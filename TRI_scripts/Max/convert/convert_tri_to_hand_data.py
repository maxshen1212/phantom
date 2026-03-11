"""
Converts TRI episode.pkl (3D hand skeletons) to Phantom HandSequence format.
Replaces the hand2d (HaMeR) step by using ARKit 3D joints directly.

Pipeline: Load episode => Transform joints to camera frame => Project to 2D =>
          Save hand_data_{left,right}.npz + video_rgb_imgs.mkv

Expects output_dir to already contain hand_det.pkl and video_L.mp4 from
convert_tri_to_epic.py. Run after bbox step in EPIC-mode pipeline.
"""

import gzip
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import mediapy as media
import numpy as np

# Reuse helpers from convert_tri_to_epic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_tri_to_epic import (
    DEFAULT_IMG_HEIGHT,
    DEFAULT_IMG_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    extract_hand_joints_3d,
    parse_to_se3,
)

logger = logging.getLogger(__name__)

# ARKit 26 joints -> MediaPipe 21 joints (matches phantom/hand.py get_list_finger_pts_from_skeleton)
# Skip: 0 (anchor), 6 (index metacarpal), 11 (middle metacarpal), 16 (ring metacarpal), 21 (little metacarpal)
# MediaPipe order: wrist(0), thumb MCP/PIP/DIP/TIP(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
ARKIT_TO_MEDIAPIPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]


# =============================================================================
# Coordinate transforms
# =============================================================================


def joints_world_to_camera(
    joints_world: np.ndarray,
    device_extrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
) -> np.ndarray:
    """
    Transform 3D joints from world frame to camera frame (vectorized).

    Args:
        joints_world: (N, 3) joint positions in world coordinates.
        device_extrinsics: 4x4 world_T_device.
        camera_extrinsics: 4x4 camera pose stored transposed in episode.pkl.

    Returns:
        (N, 3) joint positions in camera frame.
    """
    device_T_camera = np.linalg.inv(camera_extrinsics.T)
    world_T_camera = device_extrinsics @ device_T_camera
    camera_T_world = np.linalg.inv(world_T_camera)

    N = joints_world.shape[0]
    pts_h = np.hstack([joints_world, np.ones((N, 1))]).T  # (4, N)
    pts_cam = (camera_T_world @ pts_h).T[:, :3]  # (N, 3)
    return pts_cam


def joints_camera_to_pixel(
    joints_cf: np.ndarray,
    K: np.ndarray,
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
) -> np.ndarray:
    """
    Project 3D camera-frame joints to 2D pixel coordinates, scaled for target resolution.

    Args:
        joints_cf: (N, 3) joint positions in camera frame.
        K: 3x3 intrinsics at source resolution.
        src_w, src_h: Source image dimensions.
        tgt_w, tgt_h: Target image dimensions (e.g. 456x256).

    Returns:
        (N, 2) pixel coordinates (u, v) at target resolution.
    """
    K_scaled = K.copy()
    K_scaled[0, 0] *= tgt_w / src_w
    K_scaled[0, 2] *= tgt_w / src_w
    K_scaled[1, 1] *= tgt_h / src_h
    K_scaled[1, 2] *= tgt_h / src_h

    pts_h = (K_scaled @ joints_cf.T)  # (3, N)
    valid = pts_h[2, :] > 1e-6
    kpts_2d = np.zeros((joints_cf.shape[0], 2), dtype=np.float32)
    kpts_2d[valid] = (pts_h[:2, valid] / pts_h[2, valid]).T
    kpts_2d[~valid] = np.nan  # Mark invalid (behind camera) as nan
    return kpts_2d


# =============================================================================
# Per-hand and per-episode processing
# =============================================================================


def process_hand_sequence(
    episode_data: dict,
    side: str,
    num_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract hand data for one side from episode.pkl.

    Args:
        episode_data: Loaded episode dict with pose_snapshots, camera_intrinsics, camera_extrinsics.
        side: "left" or "right".
        num_frames: Number of frames (must match pose_snapshots length).

    Returns:
        (hand_detected, kpts_3d, kpts_2d)
        - hand_detected: (N,) bool
        - kpts_3d: (N, 21, 3) camera frame
        - kpts_2d: (N, 21, 2) pixel coords at TARGET resolution
    """
    pose_snapshots = episode_data["pose_snapshots"]
    camera_intrinsics_list = episode_data["camera_intrinsics"]
    camera_extrinsics_list = episode_data["camera_extrinsics"]

    hand_detected = np.zeros(num_frames, dtype=bool)
    kpts_3d = np.zeros((num_frames, 21, 3), dtype=np.float32)
    kpts_2d = np.zeros((num_frames, 21, 2), dtype=np.float32)

    for frame_idx in range(num_frames):
        pose_snapshot = pose_snapshots[frame_idx]
        hand_data = pose_snapshot.get(side)

        if hand_data is None or hand_data.get("response") is None:
            continue

        try:
            skeleton = hand_data["response"]
            device_extrinsics = parse_to_se3(skeleton.device)
            joints_world = extract_hand_joints_3d(skeleton)

            if len(joints_world) < 26:
                logger.warning("Frame %d %s hand: expected 26 joints, got %d", frame_idx, side, len(joints_world))
                continue

            # Select 21 joints (ARKit -> MediaPipe)
            joints_world_21 = joints_world[ARKIT_TO_MEDIAPIPE]  # (21, 3)

            # World -> camera frame
            K = camera_intrinsics_list[frame_idx]
            E = camera_extrinsics_list[frame_idx]
            joints_cf = joints_world_to_camera(joints_world_21, device_extrinsics, E)

            # Camera -> pixel (at target 456x256)
            joints_2d = joints_camera_to_pixel(
                joints_cf, K,
                DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT,
                TARGET_WIDTH, TARGET_HEIGHT,
            )

            hand_detected[frame_idx] = True
            kpts_3d[frame_idx] = joints_cf
            kpts_2d[frame_idx] = joints_2d

        except Exception:
            logger.exception("Failed to process frame %d %s hand", frame_idx, side)

    return hand_detected, kpts_3d, kpts_2d


def save_hand_sequences(
    left_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    right_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    output_dir: str,
) -> None:
    """
    Save HandSequence npz files in Phantom format.

    Args:
        left_data: (hand_detected, kpts_3d, kpts_2d) for left hand.
        right_data: Same for right hand.
        output_dir: Demo output directory (e.g. data/raw/tri/0/).
    """
    hand_processor_dir = os.path.join(output_dir, "hand_processor")
    os.makedirs(hand_processor_dir, exist_ok=True)

    for side, data in [("left", left_data), ("right", right_data)]:
        hand_detected, kpts_3d, kpts_2d = data
        N = len(hand_detected)
        frame_indices = np.arange(N, dtype=np.int64)

        # Replace nan in kpts_2d with 0 for downstream compatibility
        kpts_2d_safe = np.nan_to_num(kpts_2d, nan=0.0, posinf=0.0, neginf=0.0)

        path = os.path.join(hand_processor_dir, f"hand_data_{side}.npz")
        np.savez_compressed(
            path,
            hand_detected=hand_detected,
            kpts_3d=kpts_3d,
            kpts_2d=kpts_2d_safe,
            frame_indices=frame_indices,
        )
        print(f"  => {path}")


def save_video_rgb_imgs(imgs_rgb: np.ndarray, output_dir: str) -> None:
    """
    Save RGB frames as video_rgb_imgs.mkv (required by hand_inpaint).

    Args:
        imgs_rgb: (N, H, W, 3) uint8 array.
        output_dir: Demo output directory.
    """
    path = os.path.join(output_dir, "video_rgb_imgs.mkv")
    media.write_video(path, imgs_rgb, fps=10, codec="ffv1")
    print(f"  => {path}")


def process_one_demo(episode_path: str, output_dir: str) -> bool:
    """
    Process one TRI episode: produce hand_data npz files and video_rgb_imgs.mkv.

    Expects output_dir to contain:
    - hand_det.pkl (from convert_tri_to_epic)
    - video_L.mp4 (from convert_tri_to_epic, 456x256)

    Args:
        episode_path: Path to gzip-compressed episode.pkl.
        output_dir: Directory where outputs are written (same as convert_tri_to_epic).

    Returns:
        True on success, False otherwise.
    """
    video_path = os.path.join(output_dir, "video_L.mp4")
    if not os.path.exists(video_path):
        logger.error("video_L.mp4 not found in %s (run convert_tri_to_epic first)", output_dir)
        return False

    if not os.path.exists(episode_path):
        logger.error("episode.pkl not found: %s", episode_path)
        return False

    try:
        with gzip.open(episode_path, "rb") as f:
            episode_data = pickle.load(f)
    except Exception:
        logger.exception("Failed to load episode: %s", episode_path)
        return False

    num_frames = len(episode_data["pose_snapshots"])
    if num_frames == 0:
        logger.error("Empty episode: %s", episode_path)
        return False

    # Load video frames
    imgs_rgb = media.read_video(video_path)
    if len(imgs_rgb) != num_frames:
        logger.warning("Frame count mismatch: episode=%d, video=%d", num_frames, len(imgs_rgb))
        num_frames = min(num_frames, len(imgs_rgb))

    # Process both hands
    left_data = process_hand_sequence(episode_data, "left", num_frames)
    right_data = process_hand_sequence(episode_data, "right", num_frames)

    # Save outputs
    save_hand_sequences(left_data, right_data, output_dir)
    save_video_rgb_imgs(imgs_rgb[:num_frames], output_dir)

    return True


# =============================================================================
# Episode discovery and batch processing
# =============================================================================


def _sort_key_for_dir(d: str):
    try:
        return (0, int(d.split("_", 1)[1].split()[0]))
    except (IndexError, ValueError):
        return (1, d)


def find_all_episodes(base_dir: str) -> List[str]:
    """Recursively find all episode.pkl files under base_dir, sorted."""
    episode_paths = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort(key=_sort_key_for_dir)
        for fname in sorted(files):
            if fname == "episode.pkl":
                episode_paths.append(os.path.join(root, fname))
    return episode_paths


def main():
    """
    Batch-convert all TRI episodes: for each output dir with hand_det.pkl and video_L.mp4,
    produce hand_data npz files and video_rgb_imgs.mkv.

    Uses same INPUT_BASE_DIR and OUTPUT_BASE_DIR as convert_tri_to_epic.py.
    Skips episodes that already have hand_data_left.npz unless FORCE_REPROCESS.
    """
    INPUT_BASE_DIR = "/data/maxshen/Video_data/LBM_human_egocentric/egoPutKiwiInCenterOfTable/2025-11-13_12-46-27"
    OUTPUT_BASE_DIR = "/data/maxshen/phantom/data/raw/tri"
    FORCE_REPROCESS = False

    print("=" * 60)
    print("TRI to HandSequence Converter  (replaces hand2d/HaMeR)")
    print("=" * 60)
    print(f"\nSearching for episodes in: {INPUT_BASE_DIR}")

    episode_paths = find_all_episodes(INPUT_BASE_DIR)
    if not episode_paths:
        print("Error: No episode.pkl files found.")
        return 1
    print(f"Found {len(episode_paths)} episode(s).\n")

    results = {"success": 0, "skipped": 0, "failed": 0}

    for idx, episode_path in enumerate(episode_paths):
        output_dir = os.path.join(OUTPUT_BASE_DIR, str(idx))

        print("=" * 60)
        print(f"[{idx + 1}/{len(episode_paths)}] index={idx}")
        print(f"  Input : {episode_path}")
        print(f"  Output: {output_dir}")

        hand_data_left = os.path.join(output_dir, "hand_processor", "hand_data_left.npz")
        if os.path.exists(hand_data_left) and not FORCE_REPROCESS:
            print("  => Already processed, skipping.")
            results["skipped"] += 1
            continue

        if not os.path.exists(os.path.join(output_dir, "hand_det.pkl")):
            print("  => hand_det.pkl not found (run convert_tri_to_epic first), skipping.")
            results["skipped"] += 1
            continue

        success = process_one_demo(episode_path, output_dir)
        results["success" if success else "failed"] += 1

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Succeeded : {results['success']}")
    print(f"  Skipped   : {results['skipped']}")
    print(f"  Failed    : {results['failed']}")
    print(f"  Total     : {len(episode_paths)}")
    print("=" * 60)
    print()
    print("Next steps (skip hand2d, run from arm_segmentation):")
    print("  cd /data/maxshen/phantom/phantom")
    print("  python process_data.py demo_name=tri mode=arm_segmentation,action,smoothing,hand_inpaint,robot_inpaint --config-name=tri")
    print("=" * 60)

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
