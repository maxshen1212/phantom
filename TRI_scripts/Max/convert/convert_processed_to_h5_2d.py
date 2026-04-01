"""
Language-Conditioned Pure 2D HDF5 Data Parser for TRI Videos (Multiprocessing Edition).

Converts phantom-pipeline processed TRI episodes into a single HDF5 file
compatible with the BaseBufferEpicH5 dataloader.
Optimized with ProcessPoolExecutor for high-throughput processing and
incremental HDF5 writing to prevent Out-Of-Memory (OOM) on massive datasets.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
import mediapy as media
import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action layout constants (must match TRI_data_loader.py / BaseBufferEpicH5)
# ---------------------------------------------------------------------------
ACTION_DIM = 24
NUM_SUB_ACTIONS = 32
ACTION_VEC_LEN = ACTION_DIM * NUM_SUB_ACTIONS  # 768

LEFT_X2D, LEFT_Y2D = 3, 4
RIGHT_X2D, RIGHT_Y2D = 15, 16

# Camera-motion filter thresholds
TRANSLATION_THRESH_M = 0.05
ROTATION_THRESH_RAD = 0.5

# Target resolution
IMG_W, IMG_H = 456, 256

# Placeholder dimensions
STATE_DIM = 14
LANG_DIM = 768  # DistilBERT [CLS] hidden size
GMM_N_CONTACTS = 5
GMM_N_FEAT = 3
PLACEHOLDER_NOISE_SCALE = 1e-6


# ---------------------------------------------------------------------------
# Language processing  (DistilBERT + instruction dictionary)
# ---------------------------------------------------------------------------

def load_language_dict(yaml_path: str) -> Dict[str, List[str]]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    lang_dict: Dict[str, List[str]] = {}
    for task_name, entry in data.get("language_dict", {}).items():
        pool: List[str] = []
        pool.extend(entry.get("original") or [])
        pool.extend(entry.get("randomized") or [])
        if pool:
            lang_dict[task_name] = pool
    return lang_dict


def init_distilbert(device: str = "cpu") -> Tuple["DistilBertTokenizer", "DistilBertModel", str]:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.to(device)
    model.eval()
    logger.info("DistilBERT loaded on %s", device)
    return tokenizer, model, device


@torch.no_grad()
def encode_instruction(text: str, tokenizer: "DistilBertTokenizer", model: "DistilBertModel", device: str) -> np.ndarray:
    if not text:
        return np.zeros((1, LANG_DIM), dtype=np.float32)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    cls_vec = model(**inputs).last_hidden_state[:, 0, :]  # (1, 768)
    return cls_vec.cpu().numpy().astype(np.float32)


def sample_instruction(task_name: str, lang_dict: Dict[str, List[str]], rng: np.random.Generator) -> str:
    pool = lang_dict.get(task_name)
    if not pool:
        logger.warning("Task '%s' not found in language dict – using empty embedding", task_name)
        return ""
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_intrinsics(path: str) -> np.ndarray:
    with open(path) as f:
        cam = json.load(f)
    e = cam["left"]
    return np.array([
        [e["fx"], 0, e["cx"]],
        [0, e["fy"], e["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)


def load_episode_data(episode_dir: str) -> Optional[Dict]:
    dd = Path(episode_dir)
    overlay = dd / "video_overlay_Panda_shoulders.mkv"
    td_path = dd / "inpaint_processor" / "training_data_shoulders.npz"
    hl_path = dd / "hand_processor" / "hand_data_left.npz"
    hr_path = dd / "hand_processor" / "hand_data_right.npz"

    for p in [overlay, td_path, hl_path, hr_path]:
        if not p.exists():
            return None

    frames = media.read_video(str(overlay))
    td = np.load(td_path, allow_pickle=True)
    hl = np.load(hl_path, allow_pickle=True)
    hr = np.load(hr_path, allow_pickle=True)

    return {
        "frames": frames,
        "valid": td["valid"].astype(bool),
        "kpts_left": hl["kpts_2d"].astype(np.float64),
        "kpts_right": hr["kpts_2d"].astype(np.float64),
        "hand_det_left": hl["hand_detected"].astype(bool),
        "hand_det_right": hr["hand_detected"].astype(bool),
    }


# ---------------------------------------------------------------------------
# Homography computation & Filter & Action builder
# ---------------------------------------------------------------------------

def compute_pairwise_homographies(frames: np.ndarray) -> List[Optional[np.ndarray]]:
    N = len(frames)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    result: List[Optional[np.ndarray]] = []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)

    # 移除內部的 tqdm 以避免多進程時終端機輸出混亂
    for t in range(N - 1):
        curr_gray = cv2.cvtColor(frames[t + 1], cv2.COLOR_RGB2GRAY)
        curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)

        H = None
        if (prev_des is not None and curr_des is not None
                and len(prev_des) >= 4 and len(curr_des) >= 4):
            matches = bf.knnMatch(prev_des, curr_des, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) >= 4:
                src = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([curr_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        result.append(H)
        prev_kp, prev_des = curr_kp, curr_des

    return result


def _motion_from_H(H: np.ndarray, K: np.ndarray, assumed_depth: float = 0.5) -> Tuple[float, float]:
    try:
        n, Rs, Ts, _ = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return float("inf"), float("inf")

    best_rot, best_trans = float("inf"), float("inf")
    for i in range(n):
        angle = abs(np.arccos(np.clip((np.trace(Rs[i]) - 1) / 2, -1.0, 1.0)))
        tmag = np.linalg.norm(Ts[i].flatten()) * assumed_depth
        if angle < best_rot:
            best_rot, best_trans = angle, tmag
    return best_rot, best_trans


def camera_motion_mask(homographies: List[Optional[np.ndarray]], K: np.ndarray) -> np.ndarray:
    N = len(homographies) + 1
    ok = np.ones(N, dtype=bool)
    for t, H in enumerate(homographies):
        if H is None:
            ok[t + 1] = False
            continue
        rot, trans = _motion_from_H(H, K)
        if rot > ROTATION_THRESH_RAD or trans > TRANSLATION_THRESH_M:
            ok[t + 1] = False
    return ok


def _warp_pt(pt: np.ndarray, H: np.ndarray) -> np.ndarray:
    p = np.array([pt[0], pt[1], 1.0])
    q = H @ p
    return q[:2] / q[2] if abs(q[2]) > 1e-10 else pt[:2].copy()


def build_actions(kpts_L: np.ndarray, kpts_R: np.ndarray, homographies: List[Optional[np.ndarray]], valid_idx: np.ndarray, N_total: int) -> np.ndarray:
    H_inv: List[Optional[np.ndarray]] = []
    for H in homographies:
        if H is None:
            H_inv.append(None)
        else:
            try:
                H_inv.append(np.linalg.inv(H))
            except np.linalg.LinAlgError:
                H_inv.append(None)

    actions = np.zeros((len(valid_idx), ACTION_VEC_LEN), dtype=np.float32)

    for vi, t in enumerate(valid_idx):
        H_cum = np.eye(3, dtype=np.float64)
        chain_ok = True
        prev_f = t

        for k in range(NUM_SUB_ACTIONS):
            fut = min(t + k + 1, N_total - 1)
            while prev_f < fut and chain_ok:
                if prev_f < len(H_inv) and H_inv[prev_f] is not None:
                    H_cum = H_cum @ H_inv[prev_f]
                else:
                    chain_ok = False
                prev_f += 1

            wL = kpts_L[fut, 0].copy()
            wR = kpts_R[fut, 0].copy()

            if chain_ok and fut != t:
                wL = _warp_pt(wL, H_cum)
                wR = _warp_pt(wR, H_cum)

            base = k * ACTION_DIM
            actions[vi, base + LEFT_X2D] = wL[0]
            actions[vi, base + LEFT_Y2D] = wL[1]
            actions[vi, base + RIGHT_X2D] = wR[0]
            actions[vi, base + RIGHT_Y2D] = wR[1]

    return actions


# ---------------------------------------------------------------------------
# Worker & Incremental Writer for Multiprocessing
# ---------------------------------------------------------------------------

def process_episode_worker(args_tuple) -> Optional[Dict]:
    """
    Worker function executed by each CPU core.
    Processes a single episode and returns the dictionary ready for writing.
    """
    episode_dir, K, lang_embedding = args_tuple
    data = load_episode_data(episode_dir)
    if data is None:
        return None

    frames = data["frames"]
    N = len(frames)

    homographies = compute_pairwise_homographies(frames)
    cam_ok = camera_motion_mask(homographies, K)

    combined = data["valid"] & cam_ok
    valid_idx = np.where(combined)[0]

    if len(valid_idx) == 0:
        return None

    actions = build_actions(data["kpts_left"], data["kpts_right"], homographies, valid_idx, N)

    return {
        "episode_dir": episode_dir,
        "images": frames[valid_idx],
        "actions": actions,
        "lang_embedding": lang_embedding
    }


def write_single_episode_to_h5(data_grp: h5py.Group, episode_idx: int, d: Dict, rng: np.random.Generator) -> int:
    """
    Writes a single processed episode to the open HDF5 file immediately.
    Returns the number of frames written.
    """
    def _noise(shape: Tuple[int, ...]) -> np.ndarray:
        return rng.normal(0, PLACEHOLDER_NOISE_SCALE, shape).astype(np.float32)

    key = f"episode_{episode_idx}"
    dg = data_grp.create_group(key)
    N = d["actions"].shape[0]
    dg.attrs["num_samples"] = N

    dg.create_dataset("action", data=d["actions"])
    dg.create_dataset("obs/frontview_image", data=d["images"], chunks=(1, IMG_H, IMG_W, 3))
    dg.create_dataset("obs/state", data=_noise((N, STATE_DIM)))
    dg.create_dataset("obs/language_embedding", data=d["lang_embedding"])
    dg.create_dataset("contact/gmm_contacts_left", data=_noise((N, GMM_N_CONTACTS, GMM_N_FEAT)))
    dg.create_dataset("contact/gmm_contacts_right", data=_noise((N, GMM_N_CONTACTS, GMM_N_FEAT)))
    dg.create_dataset("contact/intersected_bbox_left", data=_noise((N, 4)))
    dg.create_dataset("contact/intersected_bbox_right", data=_noise((N, 4)))

    return N


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pure 2D HDF5 Data Parser for TRI videos (Multiprocessing)")

    parser.add_argument("--processed_dir", type=str, required=True, help="Root dir containing task sub-directories.")
    parser.add_argument("--intrinsics", type=str, default="/data/maxshen/phantom/phantom/camera/camera_intrinsics_tri.json", help="Phantom camera intrinsics JSON")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--lang_annotations", type=str, required=True, help="YAML file mapping task names to instruction strings")
    parser.add_argument("--device", type=str, default="cuda", help="Device for DistilBERT inference (cpu or cuda)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU cores to use for multiprocessing")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 1. Initialization
    K = load_intrinsics(args.intrinsics)
    lang_dict = load_language_dict(args.lang_annotations)
    tokenizer, bert_model, device = init_distilbert(args.device)
    lang_rng = np.random.default_rng(123)

    processed_root = Path(args.processed_dir)
    task_dirs = sorted([d for d in processed_root.iterdir() if d.is_dir()])
    logger.info("Found %d Task directories inside %s", len(task_dirs), processed_root)

    # 2. Pre-allocate tasks and pre-compute language embeddings (Fast, runs on Main Thread)
    tasks_to_run = []
    logger.info("Scanning directories and sampling language instructions...")

    for task_dir in task_dirs:
        task_name = task_dir.name
        episode_dirs = sorted(
            [d for d in task_dir.iterdir() if d.is_dir()],
            key=lambda p: (int(p.name) if p.name.isdigit() else float("inf"), p.name)
        )

        for dd in episode_dirs:
            instruction = sample_instruction(task_name, lang_dict, lang_rng)
            lang_emb = encode_instruction(instruction, tokenizer, bert_model, device)
            tasks_to_run.append((str(dd), K, lang_emb))

    logger.info("Total %d episodes queued for processing using %d workers.", len(tasks_to_run), args.num_workers)

    # 3. Multiprocessing & Incremental Writing
    total_valid_episodes = 0
    total_valid_frames = 0
    noise_rng = np.random.default_rng(42)

    # Open HDF5 file once in the main process
    with h5py.File(args.output, "w") as f_out:
        data_grp = f_out.create_group("data")

        # Start the process pool
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_episode_worker, args_tuple) for args_tuple in tasks_to_run]

            # Use tqdm to monitor progress as tasks complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Episodes"):
                try:
                    result = future.result()
                    if result is not None:
                        # Write immediately to disk and release memory
                        frames_written = write_single_episode_to_h5(data_grp, total_valid_episodes, result, noise_rng)
                        total_valid_episodes += 1
                        total_valid_frames += frames_written

                        # Explicitly delete the result dict to free RAM
                        del result
                except Exception as e:
                    logger.error("Error processing an episode: %s", e)

    logger.info("==================================================")
    logger.info("Done. %d episodes (%d frames) successfully converted and written to %s.",
                total_valid_episodes, total_valid_frames, args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
