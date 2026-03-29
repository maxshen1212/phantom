"""
Converts TRI episode.pkl (3D hand skeletons) to EPIC-KITCHENS hand_det.pkl (2D bounding boxes).
Pipeline: Load episode => Project 3D joints to 2D => Compute bbox => Save EPIC format.
Also downsamples main_camera.mp4 from 1920x1080 to 456x256 for E2FGVI compatibility.
"""

import argparse
import gzip
import json
import logging
import os
import pickle
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Kyle's directory contains tracker_pb2.py (protobuf definitions for TRI data)
tri_scripts_path = "/data/maxshen/phantom/TRI_scripts/Kyle"
if tri_scripts_path not in sys.path:
    sys.path.insert(0, tri_scripts_path)

from epic_kitchens.hoa.types import BBox, FloatVector, HandDetection, HandSide, HandState

logger = logging.getLogger(__name__)

# Original Vision Pro camera resolution
DEFAULT_IMG_WIDTH = 1920
DEFAULT_IMG_HEIGHT = 1080

# Target resolution for EPIC-KITCHENS / E2FGVI compatibility
TARGET_WIDTH = 456
TARGET_HEIGHT = 256

# Bounding box padding factor (10% expansion on each side)
BBOX_PADDING = 0.1

# ARKit hand joint names (25 joints, indices 1–25; index 0 is the anchor)
ARKIT_JOINT_NAMES = [
    "wrist",
    "thumb_knuckle",
    "thumb_intermediate_base",
    "thumb_intermediate_tip",
    "thumb_tip",
    "index_finger_metacarpal",
    "index_finger_knuckle",
    "index_finger_intermediate_base",
    "index_finger_intermediate_tip",
    "index_finger_tip",
    "middle_finger_metacarpal",
    "middle_finger_knuckle",
    "middle_finger_intermediate_base",
    "middle_finger_intermediate_tip",
    "middle_finger_tip",
    "ring_finger_metacarpal",
    "ring_finger_knuckle",
    "ring_finger_intermediate_base",
    "ring_finger_intermediate_tip",
    "ring_finger_tip",
    "little_finger_metacarpal",
    "little_finger_knuckle",
    "little_finger_intermediate_base",
    "little_finger_intermediate_tip",
    "little_finger_tip",
]


# =============================================================================
# Coordinate transform helpers
# =============================================================================


def parse_to_se3(transform) -> np.ndarray:
    """
    Convert a protobuf transform to a 4x4 SE(3) matrix.

    Args:
        transform: Protobuf object with translation (x,y,z) and rotation (quaternion x,y,z,w).

    Returns:
        4x4 numpy array (world_T_frame).
    """
    t = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
    q = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
    se3 = np.eye(4)
    se3[:3, :3] = Rotation.from_quat(q).as_matrix()
    se3[:3, 3] = t
    return se3


def extract_hand_joints_3d(skeleton_response) -> np.ndarray:
    """
    Extract up to 26 hand joint positions in world coordinates from an ARKit skeleton.

    Joint 0 is the anchor; joints 1–25 are the ARKit hand joints composed with the anchor.

    Args:
        skeleton_response: Protobuf skeleton response with .hand and .hand.hand_skeleton.

    Returns:
        (N, 3) numpy array of 3D joint positions in world coordinates (N ≤ 26).
    """
    if not hasattr(skeleton_response, "hand"):
        raise ValueError("Skeleton has no 'hand' attribute")

    hand_skeleton = skeleton_response.hand.hand_skeleton
    anchor_se3 = parse_to_se3(skeleton_response.hand.anchor_transform)

    joints_3d = [anchor_se3[:3, 3]]
    for joint_name in ARKIT_JOINT_NAMES:
        if hasattr(hand_skeleton, joint_name):
            joint_world_se3 = anchor_se3 @ parse_to_se3(getattr(hand_skeleton, joint_name))
            joints_3d.append(joint_world_se3[:3, 3])

    if len(joints_3d) < 26:
        logger.warning("Expected 26 joints, got %d", len(joints_3d))

    return np.array(joints_3d)


def project_3d_to_2d(
    points_3d: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    device_extrinsics: np.ndarray,
) -> np.ndarray:
    """
    Project 3D world points to 2D image pixel coordinates.

    camera_extrinsics is stored TRANSPOSED in episode.pkl, so we transpose back
    before inverting: device_T_camera = inv(camera_extrinsics.T).

    Args:
        points_3d: (N, 3) points in world frame.
        camera_intrinsics: 3x3 K matrix.
        camera_extrinsics: 4x4 camera pose stored transposed.
        device_extrinsics: 4x4 world_T_device.

    Returns:
        (N, 2) pixel coordinates (u, v).
    """
    device_T_camera = np.linalg.inv(camera_extrinsics.T)
    camera_T_world = np.linalg.inv(device_extrinsics @ device_T_camera)

    N = points_3d.shape[0]
    points_world_hom = np.hstack([points_3d, np.ones((N, 1))])
    points_camera = (camera_T_world @ points_world_hom.T).T[:, :3]

    # Filter out points behind the camera (z <= 0 causes flipped/invalid projections)
    valid = points_camera[:, 2] > 1e-6
    points_camera = points_camera[valid]
    if len(points_camera) == 0:
        return np.empty((0, 2))

    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    z = points_camera[:, 2]
    u = fx * (points_camera[:, 0] / z) + cx
    v = fy * (points_camera[:, 1] / z) + cy
    return np.stack([u, v], axis=1)


# =============================================================================
# Bounding box computation
# =============================================================================


def compute_bbox_from_2d_points(
    points_2d: np.ndarray,
    img_width: int,
    img_height: int,
    padding: float = BBOX_PADDING,
) -> Optional[BBox]:
    """
    Compute a normalized EPIC BBox from 2D joint coordinates.

    Args:
        points_2d: (N, 2) pixel coordinates.
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        padding: Fraction to expand the bbox on each side.

    Returns:
        BBox with coordinates in [0, 1], or None if invalid.
    """
    if len(points_2d) == 0:
        return None

    x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
    y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])

    w, h = x_max - x_min, y_max - y_min
    x_min = max(0, x_min - w * padding)
    x_max = min(img_width, x_max + w * padding)
    y_min = max(0, y_min - h * padding)
    y_max = min(img_height, y_max + h * padding)

    left, right = x_min / img_width, x_max / img_width
    top, bottom = y_min / img_height, y_max / img_height

    if left >= right or top >= bottom:
        return None

    return BBox(left=left, top=top, right=right, bottom=bottom)


# =============================================================================
# Per-hand and per-episode processing
# =============================================================================


def process_single_hand(
    hand_data: dict,
    hand_side: str,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    img_width: int,
    img_height: int,
) -> Optional[HandDetection]:
    """
    Convert one hand's skeletal data to an EPIC HandDetection.

    Args:
        hand_data: Dict with 'response' key containing a protobuf skeleton, or None.
        hand_side: 'left' or 'right'.
        camera_intrinsics: 3x3 K matrix.
        camera_extrinsics: 4x4 camera pose stored transposed.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        HandDetection, or None if data is missing or processing fails.
    """
    if hand_data is None or hand_data.get("response") is None:
        return None

    skeleton = hand_data["response"]
    try:
        device_extrinsics = parse_to_se3(skeleton.device)
        joints_3d = extract_hand_joints_3d(skeleton)
        joints_2d = project_3d_to_2d(joints_3d, camera_intrinsics, camera_extrinsics, device_extrinsics)
        bbox = compute_bbox_from_2d_points(joints_2d, img_width, img_height)
    except Exception:
        logger.exception("Failed to process %s hand", hand_side)
        return None

    if bbox is None:
        return None

    return HandDetection(
        bbox=bbox,
        score=np.float32(1.0),
        state=HandState.NO_CONTACT,
        side=HandSide.LEFT if hand_side == "left" else HandSide.RIGHT,
        object_offset=FloatVector(x=np.float32(0.0), y=np.float32(0.0)),
    )


def process_tri_episode(
    episode_path: str,
    output_dir: str,
    img_width: int = DEFAULT_IMG_WIDTH,
    img_height: int = DEFAULT_IMG_HEIGHT,
) -> Tuple[Dict[str, List[HandDetection]], np.ndarray, np.ndarray]:
    """
    Convert a TRI episode.pkl to EPIC hand_det.pkl.

    Args:
        episode_path: Path to gzip-compressed episode.pkl.
        output_dir: Directory to write hand_det.pkl.
        img_width: Original video width (default 1920).
        img_height: Original video height (default 1080).

    Returns:
        (hand_det_data, first_frame_intrinsics, first_frame_extrinsics)
    """
    print(f"Loading: {episode_path}")
    with gzip.open(episode_path, "rb") as f:
        episode_data = pickle.load(f)

    total_frames = len(episode_data["frame_timestamps"])
    duration = episode_data["frame_timestamps"][-1] - episode_data["frame_timestamps"][0]
    print(f"  frames={total_frames}, success={episode_data['success']}, duration={duration:.2f}s")

    hand_det_data: Dict[str, List[HandDetection]] = {}
    failed = {"left": 0, "right": 0}

    for frame_idx, pose_snapshot in enumerate(episode_data["pose_snapshots"]):
        K = episode_data["camera_intrinsics"][frame_idx]
        E = episode_data["camera_extrinsics"][frame_idx]
        detections = []

        for side in ["left", "right"]:
            det = process_single_hand(pose_snapshot.get(side), side, K, E, img_width, img_height)
            if det is not None:
                detections.append(det)
            elif pose_snapshot.get(side) is not None and pose_snapshot[side].get("response") is not None:
                failed[side] += 1

        hand_det_data[str(frame_idx)] = detections

    frames_with_left = sum(1 for d in hand_det_data.values() if any(x.side == HandSide.LEFT for x in d))
    frames_with_right = sum(1 for d in hand_det_data.values() if any(x.side == HandSide.RIGHT for x in d))
    empty_frames = sum(1 for d in hand_det_data.values() if len(d) == 0)
    print(f"  left={frames_with_left}, right={frames_with_right}, empty={empty_frames}")
    if failed["left"] or failed["right"]:
        print(f"  failed: left={failed['left']}, right={failed['right']}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hand_det.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(hand_det_data, f)
    print(f"  => {output_path}")

    return hand_det_data, episode_data["camera_intrinsics"][0], episode_data["camera_extrinsics"][0]


# =============================================================================
# Video downsampling
# =============================================================================


def downsample_video(
    video_src: str,
    output_dir: str,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    output_filename: str = "video_L.mp4",
) -> None:
    """
    Downsample a video to the target resolution using INTER_AREA interpolation.

    Args:
        video_src: Path to source video.
        output_dir: Output directory.
        target_w: Target width in pixels.
        target_h: Target height in pixels.
        output_filename: Output filename.
    """
    if not os.path.exists(video_src):
        print(f"  Warning: video not found: {video_src}")
        return

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"  Error: cannot open video: {video_src}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  {orig_w}x{orig_h} @ {fps:.1f}fps => {target_w}x{target_h}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA))
        frame_count += 1

    cap.release()
    writer.release()
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  => {output_path} ({frame_count} frames, {file_size:.1f} MB)")


# =============================================================================
# Camera parameter export
# =============================================================================


def save_scaled_intrinsics(
    intrinsics: np.ndarray,
    orig_w: int,
    orig_h: int,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    output_dir: str = "/data/maxshen/phantom/phantom/camera",
    output_filename: str = "camera_intrinsics_tri.json",
) -> None:
    """
    Scale camera intrinsics to match a downsampled resolution and save in Phantom format.

    Args:
        intrinsics: 3x3 original K matrix.
        orig_w, orig_h: Original video dimensions.
        target_w, target_h: Target (downsampled) dimensions.
        output_dir: Output directory.
        output_filename: Output filename.
    """
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    fx = float(intrinsics[0, 0]) * scale_x
    fy = float(intrinsics[1, 1]) * scale_y
    cx = float(intrinsics[0, 2]) * scale_x
    cy = float(intrinsics[1, 2]) * scale_y

    h_fov = 2 * np.arctan(target_w / (2 * fx)) * 180 / np.pi
    v_fov = 2 * np.arctan(target_h / (2 * fy)) * 180 / np.pi
    # Diagonal FOV uses the diagonal equivalent focal length
    f_diag = np.sqrt(fx**2 + fy**2) / np.sqrt(2)
    d_fov = 2 * np.arctan(np.sqrt(target_w**2 + target_h**2) / (2 * f_diag)) * 180 / np.pi

    cam_entry = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "disto": [0.0] * 12,
        "v_fov": float(v_fov),
        "h_fov": float(h_fov),
        "d_fov": float(d_fov),
    }
    camera_params = {"left": cam_entry, "right": dict(cam_entry)}

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(camera_params, f, indent=2)
    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  => {output_path}")


def save_extrinsics_to_json(
    extrinsics: np.ndarray = None,  # Kept for API compatibility, not used
    output_dir: str = "/data/maxshen/phantom/phantom/camera",
    output_filename: str = "camera_extrinsics_tri.json",
) -> None:
    """
    Save camera extrinsics to JSON in Phantom format.

    Reuses the EPIC ego bimanual shoulders extrinsics because TRI egocentric data
    has the same head-mounted viewing geometry. Using identity matrix would place
    reconstructed points outside the robot's reachable workspace.

    Args:
        extrinsics: Ignored (kept for backward compatibility).
        output_dir: Output directory.
        output_filename: Output filename.
    """
    src = os.path.join(output_dir, "camera_extrinsics_ego_bimanual_shoulders.json")
    dst = os.path.join(output_dir, output_filename)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied extrinsics from EPIC ego bimanual shoulders => {dst}")
        return

    # Fallback: hardcoded EPIC ego bimanual shoulders values
    extrinsics_data = [
        {
            "num_marker_seen": 114,
            "stage2_retry": 11,
            "pixel_error": 2.1157278874907863,
            "proj_func": "hand_marker_proj_world_camera",
            "camera_base_ori": [
                [-0.7220417114840215, 0.37764981440725887, 0.579686453658689],
                [0.020370475586732495, 0.8491206965938227, -0.527805917303316],
                [-0.6915495720493177, -0.3692893991088662, -0.6207934673498243],
            ],
            "camera_base_ori_rotvec": [0.2877344548443808, 2.3075097094104504, -0.6485227972051454],
            "camera_base_pos": [-0.5123627783256401, -0.11387480700266536, 0.3151264229148423],
            "p_marker_ee": [-0.041990731174163416, -0.02636865486252487, -0.01442948433864288],
            "camera_base_quat": [
                0.11139014686225811,
                0.8933022830245745,
                -0.25106152012025673,
                0.35576871621882866,
            ],
        }
    ]
    os.makedirs(output_dir, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(extrinsics_data, f, indent=2)
    print(f"  Saved hardcoded EPIC ego bimanual shoulders extrinsics => {dst}")


# =============================================================================
# Validation
# =============================================================================


def validate_output(output_path: str) -> bool:
    """
    Validate the generated hand_det.pkl file.

    Args:
        output_path: Path to hand_det.pkl.

    Returns:
        True if validation passes, False otherwise.
    """
    if not os.path.exists(output_path):
        print(f"  Error: file not found: {output_path}")
        return False

    try:
        with open(output_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict) or len(data) == 0:
            print("  Error: invalid data structure")
            return False

        print(f"  frames={len(data)}")

        # Find the first frame that has at least one detection to validate bbox
        for key, dets in data.items():
            if len(dets) > 0:
                det = dets[0]
                if not hasattr(det, "side") or not hasattr(det, "bbox"):
                    print("  Error: missing required attributes")
                    return False
                bbox = det.bbox
                if not (0 <= bbox.left < bbox.right <= 1 and 0 <= bbox.top < bbox.bottom <= 1):
                    print(f"  Error: invalid bbox in frame {key}")
                    return False
                print(f"  Sample: {det.side.name} [{bbox.left:.3f}, {bbox.top:.3f}, {bbox.right:.3f}, {bbox.bottom:.3f}]")
                break

        print("  Validation passed")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


# =============================================================================
# Episode discovery and batch processing
# =============================================================================


def _sort_key_for_dir(d: str):
    try:
        return (0, int(d.split("_", 1)[1].split()[0]))
    except (IndexError, ValueError):
        return (1, d)


def find_all_episodes(base_dir: str) -> List[str]:
    """
    Recursively find all episode.pkl files under base_dir, sorted by directory name.

    Args:
        base_dir: Root directory to search.

    Returns:
        Sorted list of absolute paths to episode.pkl files.
    """
    episode_paths = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort(key=_sort_key_for_dir)
        for fname in sorted(files):
            if fname == "episode.pkl":
                episode_paths.append(os.path.join(root, fname))
    return episode_paths


# Immediate child directories named like YYYY-MM-DD... (e.g. 2025-11-13_12-46-27)
_DATE_DIR_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _is_date_named_dir(dirname: str) -> bool:
    return bool(_DATE_DIR_PREFIX.match(dirname))


def discover_episode_pkls_ordered(input_base_dir: str) -> List[str]:
    """
    If input_base_dir contains date-stamped subfolders (YYYY-MM-DD...), collect
    episode.pkl paths in date-folder order, then episode order within each folder.
    Otherwise fall back to find_all_episodes(input_base_dir).
    """
    input_base_dir = os.path.abspath(input_base_dir)
    try:
        names = os.listdir(input_base_dir)
    except OSError:
        return find_all_episodes(input_base_dir)

    date_dirs = sorted(
        os.path.join(input_base_dir, n)
        for n in names
        if _is_date_named_dir(n) and os.path.isdir(os.path.join(input_base_dir, n))
    )
    if not date_dirs:
        return find_all_episodes(input_base_dir)

    episode_paths: List[str] = []
    for d in date_dirs:
        episode_paths.extend(find_all_episodes(d))
    return episode_paths


def load_language_dict_from_yaml(yaml_path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to load language annotations (pip install pyyaml).")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("language_dict") or {}


def resolve_language_task_key(
    explicit_key: Optional[str],
    input_base_dir: str,
    lang_dict: Dict[str, Any],
) -> Optional[str]:
    """Pick YAML language_dict key from --language-task-key or input folder basename."""
    if explicit_key:
        if explicit_key in lang_dict:
            return explicit_key
        logger.warning("language-task-key %r not found in YAML", explicit_key)
        return None

    base_path = os.path.abspath(input_base_dir.rstrip(os.sep))
    infer_path = base_path
    if _is_date_named_dir(os.path.basename(base_path)):
        parent = os.path.dirname(base_path)
        if parent:
            infer_path = parent

    base = os.path.basename(infer_path)
    candidates: List[str] = [base]
    if base.startswith("ego") and len(base) > 3:
        candidates.append(base[3:])

    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for c in ordered:
        if c in lang_dict:
            return c

    logger.warning(
        "Could not resolve language_dict key from input dir name %r (tried %s)",
        base,
        ordered,
    )
    return None


def language_entry_to_json_serializable(entry: Any) -> Any:
    """Recursively convert YAML-loaded structures to JSON-safe types."""
    if isinstance(entry, dict):
        return {k: language_entry_to_json_serializable(v) for k, v in entry.items() if v is not None}
    if isinstance(entry, list):
        return [language_entry_to_json_serializable(x) for x in entry]
    return entry


def write_language_manifest(
    output_base_dir: str,
    input_base_dir: str,
    episode_paths: List[str],
    yaml_task_key: Optional[str],
    language_dict_entry: Optional[Dict[str, Any]],
) -> str:
    os.makedirs(output_base_dir, exist_ok=True)
    manifest_path = os.path.join(output_base_dir, "language_manifest.json")
    episodes = [
        {
            "index": i,
            "episode_pkl": os.path.abspath(p),
            "source_episode_dir": os.path.abspath(os.path.dirname(p)),
        }
        for i, p in enumerate(episode_paths)
    ]
    payload = {
        "yaml_task_key": yaml_task_key,
        "language_dict_entry": language_entry_to_json_serializable(language_dict_entry)
        if language_dict_entry is not None
        else None,
        "input_base_dir": os.path.abspath(input_base_dir),
        "episodes": episodes,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return manifest_path


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(
        description="Convert TRI episode.pkl + video to EPIC-style hand_det.pkl and downsampled video."
    )
    parser.add_argument(
        "--input-base-dir",
        default=env.get("TRI_CONVERT_INPUT"),
        help="Root: single session folder or task folder with YYYY-MM-DD* children. Env: TRI_CONVERT_INPUT.",
    )
    parser.add_argument(
        "--output-base-dir",
        default=env.get("TRI_CONVERT_OUTPUT", "/data/maxshen/phantom/data/raw"),
        help="Episodes written to <this>/0, <this>/1, ... Env: TRI_CONVERT_OUTPUT.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Re-run even if hand_det.pkl exists. Env TRI_CONVERT_FORCE=1 also enables.",
    )
    parser.add_argument(
        "--language-yaml",
        default="/data/maxshen/Video_data/language_annotations.yaml",
        help="language_annotations.yaml path (language_dict).",
    )
    parser.add_argument(
        "--language-task-key",
        default=None,
        help="language_dict key (e.g. PutKiwiInCenterOfTable). If omitted, inferred from --input-base-dir basename.",
    )
    args = parser.parse_args()
    if env.get("TRI_CONVERT_FORCE", "").strip().lower() in ("1", "true", "yes"):
        args.force_reprocess = True
    if not args.input_base_dir:
        parser.error("--input-base-dir is required (or set TRI_CONVERT_INPUT).")
    return args


def process_single_episode(
    episode_path: str,
    output_dir: str,
    save_camera_params: bool = False,
) -> bool:
    """
    Process one TRI episode end-to-end.

    Generates:
    1. hand_det.pkl  - 2D bounding boxes from 3D hand skeletons
    2. video_L.mp4   - downsampled to 456x256
    3. (optional) camera_intrinsics_tri.json and camera_extrinsics_tri.json

    Args:
        episode_path: Path to episode.pkl (gzip-compressed).
        output_dir: Directory where outputs are written.
        save_camera_params: If True, write camera JSON files to Phantom camera directory.

    Returns:
        True if hand_det.pkl passes validation, False otherwise.
    """
    video_path = os.path.join(os.path.dirname(episode_path), "main_camera.mp4")

    try:
        _, first_intrinsics, first_extrinsics = process_tri_episode(episode_path, output_dir)
    except Exception:
        logger.exception("Failed to generate hand_det.pkl")
        return False

    if os.path.exists(video_path):
        try:
            downsample_video(video_path, output_dir, TARGET_WIDTH, TARGET_HEIGHT)
        except Exception:
            logger.exception("Failed to downsample video")
            return False
    else:
        print(f"  Warning: video not found at {video_path}")

    if save_camera_params:
        save_scaled_intrinsics(first_intrinsics, DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT)
        save_extrinsics_to_json(first_extrinsics)

    return validate_output(os.path.join(output_dir, "hand_det.pkl"))


def main():
    """
    Batch-convert all TRI episodes under the input root to EPIC format.

    Each episode is written to a numbered sub-directory under OUTPUT_BASE_DIR.
    Already-processed episodes (hand_det.pkl exists) are skipped unless
    force reprocess is enabled (CLI or TRI_CONVERT_FORCE).
    """
    args = parse_args()
    input_base_dir = os.path.abspath(args.input_base_dir)
    output_base_dir = os.path.abspath(args.output_base_dir)
    force_reprocess = args.force_reprocess

    print("=" * 60)
    print("TRI to EPIC Hand Detection Converter  (batch mode)")
    print("=" * 60)
    print(f"\nSearching for episodes in: {input_base_dir}")

    episode_paths = discover_episode_pkls_ordered(input_base_dir)
    if not episode_paths:
        print("Error: No episode.pkl files found.")
        return 1
    print(f"Found {len(episode_paths)} episode(s).\n")

    yaml_task_key: Optional[str] = None
    language_dict_entry: Optional[Dict[str, Any]] = None
    if yaml is None:
        logger.warning("PyYAML not installed; language_dict_entry in manifest will be null (pip install pyyaml).")
    elif os.path.isfile(args.language_yaml):
        try:
            lang_dict = load_language_dict_from_yaml(args.language_yaml)
            yaml_task_key = resolve_language_task_key(
                args.language_task_key, input_base_dir, lang_dict
            )
            if yaml_task_key is not None:
                language_dict_entry = lang_dict.get(yaml_task_key)
        except Exception:
            logger.exception("Failed to load %s; manifest will omit language_dict_entry", args.language_yaml)
    else:
        logger.warning("language-yaml not found: %s", args.language_yaml)

    results = {"success": 0, "skipped": 0, "failed": 0}
    save_camera_params_done = False

    for idx, episode_path in enumerate(episode_paths):
        output_dir = os.path.join(output_base_dir, str(idx))

        print("=" * 60)
        print(f"[{idx + 1}/{len(episode_paths)}] index={idx}")
        print(f"  Input : {episode_path}")
        print(f"  Output: {output_dir}")

        hand_det_path = os.path.join(output_dir, "hand_det.pkl")
        if os.path.exists(hand_det_path) and not force_reprocess:
            print("  => Already processed, skipping.")
            results["skipped"] += 1
            continue

        if not os.path.exists(episode_path):
            print("  ERROR: episode.pkl not found, skipping.")
            results["failed"] += 1
            continue

        save_cam = not save_camera_params_done
        success = process_single_episode(episode_path, output_dir, save_cam)

        if save_cam and success:
            save_camera_params_done = True

        results["success" if success else "failed"] += 1

    try:
        manifest_path = write_language_manifest(
            output_base_dir,
            input_base_dir,
            episode_paths,
            yaml_task_key,
            language_dict_entry,
        )
        print(f"\nWrote language manifest: {manifest_path}")
    except Exception:
        logger.exception("Failed to write language_manifest.json")

    demo_hint = os.path.basename(output_base_dir.rstrip(os.sep)) or "YOUR_DEMO_NAME"

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Succeeded : {results['success']}")
    print(f"  Skipped   : {results['skipped']}")
    print(f"  Failed    : {results['failed']}")
    print(f"  Total     : {len(episode_paths)}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  cd /data/maxshen/phantom/phantom")
    print(f"  python process_data.py demo_name={demo_hint} mode=all --config-name=tri")
    print("=" * 60)

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
