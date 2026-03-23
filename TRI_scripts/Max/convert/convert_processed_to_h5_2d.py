"""
Pure 2D HDF5 Data Parser for TRI (Toyota Research Institute) Videos.

Converts phantom-pipeline processed TRI demos into a single HDF5 file
compatible with the BaseBufferEpicH5 dataloader.

Pipeline per demo:
  1. Read overlay video frames and 2D hand keypoints
  2. Compute pairwise homographies via SIFT + RANSAC
  3. Filter frames by camera motion (translation > 5cm or rotation > 0.5 rad)
  4. Warp future 2D wrist keypoints back to current frame via cumulative homography
  5. Pack into 32x24 action vectors and write HDF5

Usage:
    python convert_processed_to_h5_2d.py \
        --processed_dir /data/maxshen/phantom/data/processed/tri \
        --intrinsics /data/maxshen/phantom/phantom/camera/camera_intrinsics_tri.json \
        --output /data/maxshen/phantom/data/tri_2d.h5
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import mediapy as media
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action layout constants (must match TRI_data_loader.py / BaseBufferEpicH5)
#
# Each sub-action has 24 values:
#   Left  hand: x(0) y(1) z(2) x2d(3) y2d(4) r1-r6(5-10) g(11)
#   Right hand: x(12) y(13) z(14) x2d(15) y2d(16) r1-r6(17-22) g(23)
#
# 32 sub-actions total => flat vector of 768.
# BaseBufferEpicH5 subsamples with stride 4 => 8 waypoints x 4 values = 32.
# ---------------------------------------------------------------------------
ACTION_DIM = 24
NUM_SUB_ACTIONS = 32
ACTION_VEC_LEN = ACTION_DIM * NUM_SUB_ACTIONS  # 768

LEFT_X2D, LEFT_Y2D = 3, 4
RIGHT_X2D, RIGHT_Y2D = 15, 16

# Camera-motion filter thresholds
TRANSLATION_THRESH_M = 0.05
ROTATION_THRESH_RAD = 0.5

# Target resolution (post-phantom-pipeline overlay video)
IMG_W, IMG_H = 456, 256

# Placeholder dimensions for fields the 2D pipeline does not produce
STATE_DIM = 14
LANG_DIM = 512
GMM_N_CONTACTS = 5
GMM_N_FEAT = 3


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_intrinsics(path: str) -> np.ndarray:
    """Load 3x3 camera intrinsic matrix from Phantom camera JSON."""
    with open(path) as f:
        cam = json.load(f)
    e = cam["left"]
    return np.array([
        [e["fx"], 0, e["cx"]],
        [0, e["fy"], e["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)


def load_demo_data(demo_dir: str) -> Optional[Dict]:
    """Load all required files for a single demo. Returns None on failure."""
    dd = Path(demo_dir)
    overlay = dd / "video_overlay_Panda_shoulders.mkv"
    td_path = dd / "inpaint_processor" / "training_data_shoulders.npz"
    hl_path = dd / "hand_processor" / "hand_data_left.npz"
    hr_path = dd / "hand_processor" / "hand_data_right.npz"

    for p in [overlay, td_path, hl_path, hr_path]:
        if not p.exists():
            logger.warning("Missing %s – skipping demo %s", p.name, dd.name)
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
# Homography computation
# ---------------------------------------------------------------------------

def compute_pairwise_homographies(
    frames: np.ndarray,
) -> List[Optional[np.ndarray]]:
    """
    Compute H[t] that maps a point in frame t to frame t+1 for every
    consecutive pair, using SIFT features with ratio-test + RANSAC.
    """
    N = len(frames)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    result: List[Optional[np.ndarray]] = []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)

    for t in tqdm(range(N - 1), desc="  Homographies", leave=False):
        curr_gray = cv2.cvtColor(frames[t + 1], cv2.COLOR_RGB2GRAY)
        curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)

        H = None
        if (prev_des is not None and curr_des is not None
                and len(prev_des) >= 4 and len(curr_des) >= 4):
            matches = bf.knnMatch(prev_des, curr_des, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) >= 4:
                src = np.float32(
                    [prev_kp[m.queryIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                dst = np.float32(
                    [curr_kp[m.trainIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        result.append(H)
        prev_kp, prev_des = curr_kp, curr_des

    return result


# ---------------------------------------------------------------------------
# Camera-motion filter
# ---------------------------------------------------------------------------

def _motion_from_H(
    H: np.ndarray, K: np.ndarray, assumed_depth: float = 0.5,
) -> Tuple[float, float]:
    """Estimate (rotation_rad, translation_m) from a homography."""
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


def camera_motion_mask(
    homographies: List[Optional[np.ndarray]],
    K: np.ndarray,
) -> np.ndarray:
    """
    Boolean mask (N,) where True = frame passes the camera-motion filter.

    Frame t+1 is rejected if the transition H[t] (frame t -> t+1) exceeds
    the rotation or translation threshold.  Frame 0 is always valid.
    """
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


# ---------------------------------------------------------------------------
# Action builder  (cumulative homography + wrist warping)
# ---------------------------------------------------------------------------

def _warp_pt(pt: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply homography H to a single 2D point."""
    p = np.array([pt[0], pt[1], 1.0])
    q = H @ p
    return q[:2] / q[2] if abs(q[2]) > 1e-10 else pt[:2].copy()


def build_actions(
    kpts_L: np.ndarray,
    kpts_R: np.ndarray,
    homographies: List[Optional[np.ndarray]],
    valid_idx: np.ndarray,
    N_total: int,
) -> np.ndarray:
    """
    Build the (len(valid_idx), 768) action array.

    For each valid frame t and each sub-action k = 0..31:
      future = min(t + k + 1, N_total - 1)
      Warp the wrist keypoint at `future` back to frame t's viewpoint
      using the cumulative inverse homography chain.

    H_{future -> t} = inv(H[t]) @ inv(H[t+1]) @ ... @ inv(H[future-1])
    """
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

    for vi, t in enumerate(tqdm(valid_idx, desc="  Actions", leave=False)):
        H_cum = np.eye(3, dtype=np.float64)
        chain_ok = True
        prev_f = t

        for k in range(NUM_SUB_ACTIONS):
            fut = min(t + k + 1, N_total - 1)

            # Extend the cumulative inverse-homography chain one step at a time
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
# HDF5 writer  (BaseBufferEpicH5-compatible)
# ---------------------------------------------------------------------------

def write_h5(output_path: str, demos: List[Dict]) -> None:
    """
    Write all processed demos into a single HDF5 file.

    Layout per demo (matches BaseBufferEpicH5._sample expectations):
        data/<key>/
            attrs:  num_samples
            action:                      (N, 768)   float32
            obs/frontview_image:         (N, H, W, 3)  uint8
            obs/state:                   (N, 14)    float32  (placeholder)
            obs/language_embedding:      (1, 512)   float32  (placeholder)
            contact/gmm_contacts_left:   (N, 5, 3)  float32  (placeholder)
            contact/gmm_contacts_right:  (N, 5, 3)  float32  (placeholder)
            contact/intersected_bbox_left:  (N, 4)  float32  (placeholder)
            contact/intersected_bbox_right: (N, 4)  float32  (placeholder)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")

        for i, d in enumerate(demos):
            key = f"demo_{i}"
            dg = data_grp.create_group(key)
            N = d["actions"].shape[0]
            dg.attrs["num_samples"] = N

            dg.create_dataset("action", data=d["actions"])
            dg.create_dataset(
                "obs/frontview_image", data=d["images"],
                chunks=(1, IMG_H, IMG_W, 3),
            )
            dg.create_dataset(
                "obs/state",
                data=np.zeros((N, STATE_DIM), dtype=np.float32),
            )
            dg.create_dataset(
                "obs/language_embedding",
                data=np.zeros((1, LANG_DIM), dtype=np.float32),
            )
            dg.create_dataset(
                "contact/gmm_contacts_left",
                data=np.zeros((N, GMM_N_CONTACTS, GMM_N_FEAT), dtype=np.float32),
            )
            dg.create_dataset(
                "contact/gmm_contacts_right",
                data=np.zeros((N, GMM_N_CONTACTS, GMM_N_FEAT), dtype=np.float32),
            )
            dg.create_dataset(
                "contact/intersected_bbox_left",
                data=np.zeros((N, 4), dtype=np.float32),
            )
            dg.create_dataset(
                "contact/intersected_bbox_right",
                data=np.zeros((N, 4), dtype=np.float32),
            )

    total_frames = sum(d["actions"].shape[0] for d in demos)
    logger.info(
        "Wrote %d demos (%d total frames) => %s",
        len(demos), total_frames, output_path,
    )


# ---------------------------------------------------------------------------
# Per-demo orchestrator
# ---------------------------------------------------------------------------

def process_demo(demo_dir: str, K: np.ndarray) -> Optional[Dict]:
    """Process one demo directory. Returns dict with 'images' and 'actions'."""
    data = load_demo_data(demo_dir)
    if data is None:
        return None

    frames = data["frames"]
    N = len(frames)
    logger.info(
        "  Frames: %d | pipeline-valid: %d/%d",
        N, data["valid"].sum(), N,
    )

    homographies = compute_pairwise_homographies(frames)

    cam_ok = camera_motion_mask(homographies, K)
    logger.info("  Camera-motion valid: %d/%d", cam_ok.sum(), N)

    combined = data["valid"] & cam_ok
    valid_idx = np.where(combined)[0]
    logger.info("  Combined valid: %d/%d", len(valid_idx), N)

    if len(valid_idx) == 0:
        logger.warning("  No valid frames – skipping.")
        return None

    actions = build_actions(
        data["kpts_left"], data["kpts_right"],
        homographies, valid_idx, N,
    )

    return {"images": frames[valid_idx], "actions": actions}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pure 2D HDF5 Data Parser for TRI videos",
    )
    parser.add_argument(
        "--processed_dir", type=str,
        default="/data/maxshen/phantom/data/processed/tri",
        help="Root dir containing numbered demo sub-directories",
    )
    parser.add_argument(
        "--intrinsics", type=str,
        default="/data/maxshen/phantom/phantom/camera/camera_intrinsics_tri.json",
        help="Phantom camera intrinsics JSON",
    )
    parser.add_argument(
        "--output", type=str,
        default="/data/maxshen/phantom/data/tri_2d.h5",
        help="Output HDF5 path",
    )
    args = parser.parse_args()

    K = load_intrinsics(args.intrinsics)
    logger.info("Camera K:\n%s", K)

    processed = Path(args.processed_dir)
    demo_dirs = sorted(
        [d for d in processed.iterdir() if d.is_dir()],
        key=lambda p: (int(p.name) if p.name.isdigit() else float("inf"), p.name),
    )
    logger.info("Found %d demo directories", len(demo_dirs))

    demos: List[Dict] = []
    for dd in demo_dirs:
        logger.info("=" * 50)
        logger.info("Processing demo %s …", dd.name)
        result = process_demo(str(dd), K)
        if result is not None:
            demos.append(result)

    if not demos:
        logger.error("No demos produced valid data.")
        return 1

    write_h5(args.output, demos)

    logger.info("=" * 50)
    logger.info("Done. %d demos converted.", len(demos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
