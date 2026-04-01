"""Convert TRI LBM_sim_egocentric data to LeRobot v3 dataset for Diffusion Policy.

On NFS or slow storage, decompressing every RGB stream from ``observations.npz``
is the dominant cost (e.g. tens of minutes per episode for four cameras). **By
default we load only the main egocentric scene camera** ``scene_right_0``
(Kyle's first-person view). That cuts I/O and RAM versus multi-camera loads and
speeds Phase 1 iteration; Masquerade-style pretraining also emphasizes this
view, so early experiments often skip wrist cameras. Add more views with
``--cameras ...`` or use ``--all-cameras`` when you need the full set.

Usage:
    python convert_tri_to_lerobot.py \
        --input_dir /data/maxshen/Video_data/LBM_sim_egocentric/BimanualPlaceAppleFromBowlOnCuttingBoard \
        --output_repo tri/lbm_sim_ego_full \
        --output_dir /data/maxshen/lerobot_output_full \
        --fps 10

    # Quick test (1 episode):
    python convert_tri_to_lerobot.py \
        --input_dir /data/maxshen/Video_data/LBM_sim_egocentric/BimanualPlaceAppleFromBowlOnCuttingBoard \
        --output_repo tri/lbm_sim_ego \
        --output_dir /data/maxshen/lerobot_output \
        --fps 10 \
        --max_episodes 1

Video codec:
    Default ``--vcodec h264`` (CPU software H.264 via FFmpeg). Avoid ``libsvtav1``
    (very slow). On SLURM, ``h264_nvenc`` may fail if NVIDIA drivers are too old.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default: single main egocentric RGB (minimal NFS I/O; see module docstring).
DEFAULT_CAMERAS: tuple[str, ...] = ("scene_right_0",)

EGO_STATE_KEYS: list[tuple[str, int]] = [
    ("robot__actual__poses__right::panda__xyz", 3),
    ("robot__actual__poses__right::panda__rot_6d", 6),
    ("robot__actual__grippers__right::panda_hand", 1),
    ("robot__actual__poses__left::panda__xyz", 3),
    ("robot__actual__poses__left::panda__rot_6d", 6),
    ("robot__actual__grippers__left::panda_hand", 1),
]
STATE_DIM = sum(d for _, d in EGO_STATE_KEYS)
ACTIONS_KEY = "actions"


def npz_array_shape(path: Path, key: str) -> tuple:
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(key + ".npy") as f:
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
    return shape


def discover_episodes(input_dir: Path) -> list[Path]:
    episodes = sorted(
        p.parent
        for p in input_dir.rglob("observations.npz")
        if p.parent.name == "processed"
    )
    logger.info(f"Discovered {len(episodes)} episodes under {input_dir}")
    return episodes


def task_name_from_episode(processed_dir: Path) -> str:
    meta_path = processed_dir / "metadata.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text())
        skills = meta.get("skills")
        if skills and isinstance(skills, dict):
            return next(iter(skills))
    parts = processed_dir.parts
    for i, part in enumerate(parts):
        if part == "LBM_sim_egocentric" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown_task"


def resolve_camera_entries(
    processed_dir: Path, semantic_names: list[str] | None
) -> dict[str, str]:
    """hardware_id -> semantic name. ``semantic_names is None`` means every camera in metadata."""
    cam_map: dict[str, str] = yaml.safe_load(
        (processed_dir / "metadata.yaml").read_text()
    )["camera_id_to_semantic_name"]
    if semantic_names is None:
        return dict(cam_map)
    # one hardware id per semantic label (first match in yaml order)
    by_sem: dict[str, str] = {}
    for hw_id, sem in cam_map.items():
        if sem not in by_sem:
            by_sem[sem] = hw_id
    out: dict[str, str] = {}
    for sem in semantic_names:
        if sem not in by_sem:
            avail = sorted(set(cam_map.values()))
            raise KeyError(f"Unknown camera {sem!r}; available: {avail}")
        out[by_sem[sem]] = sem
    return out


def convert_episode(
    dataset,
    processed_dir: Path,
    episode_idx: int,
    camera_names: list[str] | None,
) -> int:
    obs_path = processed_dir / "observations.npz"
    act_path = processed_dir / "actions.npz"

    task = task_name_from_episode(processed_dir)
    cam_entries = resolve_camera_entries(processed_dir, camera_names)

    act = np.load(str(act_path), allow_pickle=True)
    actions = act[ACTIONS_KEY].astype(np.float32)
    T = actions.shape[0]

    logger.info(f"  Episode {episode_idx}: loading state arrays...")
    state_arrays = []
    obs_lazy = np.load(str(obs_path), allow_pickle=True)
    for key, expected_dim in EGO_STATE_KEYS:
        arr = obs_lazy[key]
        assert arr.shape == (T, expected_dim), f"{key}: expected ({T},{expected_dim}), got {arr.shape}"
        state_arrays.append(arr.astype(np.float32))

    state_all = np.concatenate(state_arrays, axis=1)
    assert state_all.shape == (T, STATE_DIM)

    logger.info(
        f"  Episode {episode_idx}: T={T}, task='{task}', "
        f"cameras={list(cam_entries.values())}. Loading camera arrays..."
    )

    cam_arrays: dict[str, np.ndarray] = {}
    for cid, sem_name in cam_entries.items():
        logger.info(f"    Loading camera {sem_name} ({cid})...")
        cam_arrays[sem_name] = obs_lazy[cid]
        assert cam_arrays[sem_name].shape[0] == T, (
            f"camera {sem_name} T={cam_arrays[sem_name].shape[0]} != {T}"
        )

    logger.info(f"  Writing {T} frames...")

    for t in range(T):
        frame: dict = {"task": task}
        frame["observation.state"] = np.asarray(state_all[t], dtype=np.float32).reshape(-1)
        frame["action"] = np.asarray(actions[t], dtype=np.float32).reshape(-1)
        # HWC uint8/float numpy per frame; LeRobot image_writer accepts HWC without permute.
        for sem_name, img_arr in cam_arrays.items():
            frame[f"observation.images.{sem_name}"] = img_arr[t]
        dataset.add_frame(frame)

    dataset.save_episode()

    del cam_arrays, obs_lazy
    logger.info(f"  Episode {episode_idx}: saved {T} frames.")
    return T


def build_features(
    action_dim: int,
    cam_entries: dict[str, str],
    img_h: int,
    img_w: int,
    use_videos: bool,
) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {
                "axes": [
                    "ego_left_xyz_x", "ego_left_xyz_y", "ego_left_xyz_z",
                    "ego_left_rot6d_0", "ego_left_rot6d_1", "ego_left_rot6d_2",
                    "ego_left_rot6d_3", "ego_left_rot6d_4", "ego_left_rot6d_5",
                    "ego_left_gripper",
                    "ego_right_xyz_x", "ego_right_xyz_y", "ego_right_xyz_z",
                    "ego_right_rot6d_0", "ego_right_rot6d_1", "ego_right_rot6d_2",
                    "ego_right_rot6d_3", "ego_right_rot6d_4", "ego_right_rot6d_5",
                    "ego_right_gripper",
                ],
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
    }

    img_dtype = "video" if use_videos else "image"
    for sem_name in cam_entries.values():
        features[f"observation.images.{sem_name}"] = {
            "dtype": img_dtype,
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        }

    return features


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TRI LBM_sim data to LeRobot v3")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root of TRI data (e.g. .../LBM_sim_egocentric)")
    parser.add_argument("--output_repo", type=str, required=True,
                        help="LeRobot repo_id (e.g. tri/lbm_sim_ego)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Local root for the output dataset")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit number of episodes (for testing)")
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            f"Semantic names to load, e.g. wrist_right_minus. "
            f"Default (omit flag): {list(DEFAULT_CAMERAS)}. "
            "Use --all-cameras for every camera in metadata.yaml."
        ),
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Load all cameras from metadata (heavy on NFS).",
    )
    parser.add_argument("--use_videos", action="store_true", default=True,
                        help="Store images as MP4 video tracks (default)")
    parser.add_argument("--no_videos", dest="use_videos", action="store_false",
                        help="Store images as PNGs instead of MP4")
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help=(
            "Video codec (LeRobot token). Default h264 (CPU). "
            "Avoid h264_nvenc on clusters with outdated NVIDIA drivers. "
            "Other: hevc, libsvtav1 (slow), auto, h264_nvenc, ..."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.all_cameras:
        camera_names: list[str] | None = None
    elif args.cameras:
        camera_names = list(args.cameras)
    else:
        camera_names = list(DEFAULT_CAMERAS)
    vcodec = args.vcodec

    episodes = discover_episodes(input_dir)
    if not episodes:
        logger.error("No episodes found. Check --input_dir.")
        sys.exit(1)

    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
        logger.info(f"Limited to {len(episodes)} episodes")

    first_ep = episodes[0]
    try:
        cam_entries = resolve_camera_entries(first_ep, camera_names)
    except KeyError as e:
        logger.error(f"Camera selection failed on probe episode {first_ep}: {e}")
        sys.exit(1)

    first_cam_id = next(iter(cam_entries))
    img_shape = npz_array_shape(first_ep / "observations.npz", first_cam_id)
    _, img_h, img_w, img_c = img_shape
    assert img_c == 3, f"Expected 3-channel RGB, got {img_c}"

    act_shape = npz_array_shape(first_ep / "actions.npz", ACTIONS_KEY)
    action_dim = act_shape[1]

    logger.info(f"Image shape: ({img_h}, {img_w}, 3), Action dim: {action_dim}")
    logger.info(f"Cameras: {list(cam_entries.values())}")
    logger.info(f"State dim: {STATE_DIM} (ego-flipped 20D)")
    logger.info(f"Video codec: {vcodec}")

    features = build_features(action_dim, cam_entries, img_h, img_w, args.use_videos)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.output_repo,
        fps=args.fps,
        features=features,
        root=str(output_dir),
        use_videos=args.use_videos,
        vcodec=vcodec,
        robot_type="panda_bimanual",
    )

    total_frames = 0
    failed = 0
    for idx, ep_dir in enumerate(episodes):
        logger.info(f"[{idx + 1}/{len(episodes)}] Converting {ep_dir}")
        try:
            t = convert_episode(dataset, ep_dir, idx, camera_names)
            total_frames += t
        except Exception:
            logger.error("Episode failed, skipping path=%s", ep_dir)
            logger.exception("convert_episode error (continuing with next episode)")
            failed += 1
            continue

    dataset.finalize()
    logger.info(
        f"Done. {len(episodes) - failed}/{len(episodes)} episodes OK, "
        f"{failed} failed, {total_frames} total frames -> {output_dir}"
    )


if __name__ == "__main__":
    main()
