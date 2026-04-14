# -*- coding: utf-8 -*-
"""Convert TRI LBM_sim_egocentric episodes to LeRobot v3.

This script only remaps data format:
- No frame/geometry transforms are applied.
- observation.state is built by concatenating fixed keys in EGO_STATE_KEYS.
- action is copied from actions.npz.
- selected camera streams are copied per frame.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

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


def trim_to_length(
    arr: np.ndarray, expected_t: int, label: str, episode_idx: int
) -> np.ndarray:
    """Trim T+1 arrays to T, or raise if length is inconsistent."""
    if arr.shape[0] == expected_t + 1:
        logger.warning(
            "  Episode %s: %s has T+1 rows; trimming to T.", episode_idx, label
        )
        return arr[:expected_t]
    if arr.shape[0] != expected_t:
        raise ValueError(
            f"{label}: expected T={expected_t}, got {arr.shape[0]}"
        )
    return arr


def npz_array_shape(path: Path, key: str) -> tuple:
    """Return array shape from an npz key without loading full data."""
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(key + ".npy") as f:
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
    return shape


def load_camera_map(processed_dir: Path) -> dict[str, str]:
    """Return camera_id_to_semantic_name from metadata.yaml."""
    metadata = yaml.safe_load((processed_dir / "metadata.yaml").read_text())
    return metadata["camera_id_to_semantic_name"]


def load_actions(
    act_path: Path, expected_action_dim: int | None = None
) -> np.ndarray:
    """Load actions as float32 and validate shape."""
    with np.load(str(act_path), allow_pickle=True) as act:
        actions = act[ACTIONS_KEY].astype(np.float32)

    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D, got shape={actions.shape}")

    if (
        expected_action_dim is not None
        and actions.shape[1] != expected_action_dim
    ):
        raise ValueError(
            f"action dim mismatch: expected {expected_action_dim}, got {actions.shape[1]}"
        )

    return actions


def validate_state_shapes(obs_path: Path, t: int) -> None:
    """Validate all state arrays against expected dims and T/T+1 length."""
    for key, expected_dim in EGO_STATE_KEYS:
        state_shape = npz_array_shape(obs_path, key)
        if len(state_shape) != 2:
            raise ValueError(f"{key} must be 2D, got shape={state_shape}")
        if state_shape[1] != expected_dim:
            raise ValueError(
                f"{key} expected dim {expected_dim}, got {state_shape[1]}"
            )
        if state_shape[0] not in (t, t + 1):
            raise ValueError(
                f"{key} expected T or T+1 rows ({t}/{t + 1}), got {state_shape[0]}"
            )


def validate_camera_shapes(
    obs_path: Path, cam_entries: dict[str, str], t: int
) -> tuple[int, int]:
    """Validate selected camera arrays and return shared image (H, W)."""
    image_hw: tuple[int, int] | None = None

    for cid, sem_name in cam_entries.items():
        cam_shape = npz_array_shape(obs_path, cid)
        if len(cam_shape) != 4:
            raise ValueError(
                f"camera {sem_name} must be 4D [T,H,W,C], got shape={cam_shape}"
            )
        if cam_shape[3] != 3:
            raise ValueError(
                f"camera {sem_name} expected RGB C=3, got {cam_shape[3]}"
            )
        if cam_shape[0] not in (t, t + 1):
            raise ValueError(
                f"camera {sem_name} expected T or T+1 rows ({t}/{t + 1}), got {cam_shape[0]}"
            )

        hw = (int(cam_shape[1]), int(cam_shape[2]))
        if image_hw is None:
            image_hw = hw
        elif hw != image_hw:
            raise ValueError(
                f"camera shape mismatch within episode: expected {image_hw}, got {hw} ({sem_name})"
            )

    if image_hw is None:
        raise ValueError("no cameras resolved for episode")

    return image_hw


def discover_episodes(input_dir: Path) -> list[Path]:
    """Find all processed episode directories under input_dir."""
    episodes = sorted(
        p.parent
        for p in input_dir.rglob("observations.npz")
        if p.parent.name == "processed"
    )
    logger.info("Discovered %s episodes under %s", len(episodes), input_dir)
    return episodes


def task_name_from_episode(processed_dir: Path) -> str:
    """Read task name from metadata.yaml; fallback to path-derived name."""
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
    """Map hardware camera IDs to selected semantic names for one episode."""
    cam_map = load_camera_map(processed_dir)
    if semantic_names is None:
        return dict(cam_map)

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


def preflight_episodes(
    episodes: list[Path],
    camera_names: list[str] | None,
    strict: bool,
) -> tuple[list[Path], int, tuple[int, int]]:
    """Validate episode files/keys/shapes and return valid episodes plus shared dims."""
    valid_episodes: list[Path] = []
    action_dim_ref: int | None = None
    image_hw_ref: tuple[int, int] | None = None
    failures: list[tuple[Path, str]] = []

    for ep_dir in episodes:
        try:
            obs_path = ep_dir / "observations.npz"
            act_path = ep_dir / "actions.npz"
            meta_path = ep_dir / "metadata.yaml"

            if not obs_path.exists():
                raise FileNotFoundError("missing observations.npz")
            if not act_path.exists():
                raise FileNotFoundError("missing actions.npz")
            if not meta_path.exists():
                raise FileNotFoundError("missing metadata.yaml")

            cam_entries = resolve_camera_entries(ep_dir, camera_names)

            act_shape = npz_array_shape(act_path, ACTIONS_KEY)
            if len(act_shape) != 2:
                raise ValueError(f"actions must be 2D, got shape={act_shape}")
            t, action_dim = int(act_shape[0]), int(act_shape[1])

            if action_dim_ref is None:
                action_dim_ref = action_dim
            elif action_dim != action_dim_ref:
                raise ValueError(
                    f"action dim mismatch: expected {action_dim_ref}, got {action_dim}"
                )

            validate_state_shapes(obs_path, t)
            ep_image_hw = validate_camera_shapes(obs_path, cam_entries, t)

            if image_hw_ref is None:
                image_hw_ref = ep_image_hw
            elif ep_image_hw != image_hw_ref:
                raise ValueError(
                    f"image shape mismatch across episodes: expected {image_hw_ref}, got {ep_image_hw}"
                )

            valid_episodes.append(ep_dir)
        except Exception as e:
            failures.append((ep_dir, str(e)))
            logger.warning("Preflight failed for %s: %s", ep_dir, e)

    logger.info(
        "Preflight summary: total=%s, valid=%s, failed=%s",
        len(episodes),
        len(valid_episodes),
        len(failures),
    )

    if failures and strict:
        logger.error("Strict preflight is enabled; aborting due to failures.")
        for ep_dir, reason in failures[:10]:
            logger.error("  %s -> %s", ep_dir, reason)
        raise RuntimeError("strict preflight failed")

    if not valid_episodes:
        raise RuntimeError("no valid episodes after preflight")

    if action_dim_ref is None or image_hw_ref is None:
        raise RuntimeError(
            "preflight could not determine shared action/image dims"
        )

    return valid_episodes, action_dim_ref, image_hw_ref


def convert_episode(
    dataset,
    processed_dir: Path,
    episode_idx: int,
    camera_names: list[str] | None,
    expected_action_dim: int,
) -> int:
    """Convert one processed episode directory and return its frame count."""
    obs_path = processed_dir / "observations.npz"
    act_path = processed_dir / "actions.npz"

    task = task_name_from_episode(processed_dir)
    cam_entries = resolve_camera_entries(processed_dir, camera_names)

    actions = load_actions(act_path, expected_action_dim=expected_action_dim)
    t = actions.shape[0]

    logger.info("  Episode %s: loading state arrays...", episode_idx)
    state_arrays: list[np.ndarray] = []
    cam_arrays: dict[str, np.ndarray] = {}

    with np.load(str(obs_path), allow_pickle=True) as obs_lazy:
        for key, expected_dim in EGO_STATE_KEYS:
            arr = trim_to_length(obs_lazy[key], t, key, episode_idx)
            if arr.shape[1] != expected_dim:
                raise ValueError(
                    f"{key}: expected dim {expected_dim}, got {arr.shape[1]}"
                )
            state_arrays.append(arr.astype(np.float32))

        state_all = np.concatenate(state_arrays, axis=1)
        assert state_all.shape == (t, STATE_DIM)

        logger.info(
            "  Episode %s: T=%s, task=%r, cameras=%s. Loading camera arrays...",
            episode_idx,
            t,
            task,
            list(cam_entries.values()),
        )

        for cid, sem_name in cam_entries.items():
            logger.info("    Loading camera %s (%s)...", sem_name, cid)
            cam_arr = trim_to_length(
                obs_lazy[cid], t, f"camera {sem_name}", episode_idx
            )
            cam_arrays[sem_name] = cam_arr

    logger.info("  Writing %s frames...", t)
    for i in range(t):
        frame = {
            "task": task,
            "observation.state": np.asarray(
                state_all[i], dtype=np.float32
            ).reshape(-1),
            "action": np.asarray(actions[i], dtype=np.float32).reshape(-1),
        }
        for sem_name, img_arr in cam_arrays.items():
            frame[f"observation.images.{sem_name}"] = img_arr[i]
        dataset.add_frame(frame)

    dataset.save_episode()
    logger.info("  Episode %s: saved %s frames.", episode_idx, t)
    return t


def build_features(
    action_dim: int, cam_entries: dict[str, str], img_h: int, img_w: int
) -> dict:
    """Build LeRobot feature schema for state, action, and selected cameras."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {
                "axes": [
                    "ego_right_xyz_x",
                    "ego_right_xyz_y",
                    "ego_right_xyz_z",
                    "ego_right_rot6d_0",
                    "ego_right_rot6d_1",
                    "ego_right_rot6d_2",
                    "ego_right_rot6d_3",
                    "ego_right_rot6d_4",
                    "ego_right_rot6d_5",
                    "ego_right_gripper",
                    "ego_left_xyz_x",
                    "ego_left_xyz_y",
                    "ego_left_xyz_z",
                    "ego_left_rot6d_0",
                    "ego_left_rot6d_1",
                    "ego_left_rot6d_2",
                    "ego_left_rot6d_3",
                    "ego_left_rot6d_4",
                    "ego_left_rot6d_5",
                    "ego_left_gripper",
                ],
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
    }

    for sem_name in cam_entries.values():
        features[f"observation.images.{sem_name}"] = {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        }

    return features


def post_conversion_check(
    repo_id: str, root: Path, required_feature_keys: set[str]
) -> None:
    """Load converted dataset and verify key metadata is present."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    check_ds = LeRobotDataset(
        repo_id=repo_id, root=str(root), download_videos=False
    )

    available_keys = set(check_ds.meta.features.keys())
    missing = sorted(required_feature_keys - available_keys)
    if missing:
        raise RuntimeError(f"post-check failed: missing feature keys {missing}")
    if len(check_ds) == 0:
        raise RuntimeError("post-check failed: dataset has zero frames")

    logger.info(
        "Post-check OK: episodes=%s, frames=%s",
        len(check_ds.meta.episodes),
        len(check_ds),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TRI LBM_sim data to LeRobot v3"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Root of TRI data"
    )
    parser.add_argument(
        "--output_repo", type=str, required=True, help="LeRobot repo_id"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Local output dataset dir"
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit episodes for testing",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=None,
        metavar="NAME",
        help=f"Semantic camera names; default: {list(DEFAULT_CAMERAS)}",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="auto",
        help="Codec token. e.g. auto, h264, libsvtav1, hevc, h264_nvenc",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    episodes = discover_episodes(input_dir)
    if not episodes:
        logger.error("No episodes found. Check --input_dir.")
        sys.exit(1)

    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
        logger.info("Limited to %s episodes", len(episodes))

    camera_names = list(args.cameras) if args.cameras else list(DEFAULT_CAMERAS)

    try:
        episodes, action_dim, (img_h, img_w) = preflight_episodes(
            episodes,
            camera_names,
            strict=True,
        )
    except Exception as e:
        logger.error("Preflight failed: %s", e)
        sys.exit(1)

    first_ep = episodes[0]
    try:
        cam_entries = resolve_camera_entries(first_ep, camera_names)
    except KeyError as e:
        logger.error(
            "Camera selection failed on probe episode %s: %s", first_ep, e
        )
        sys.exit(1)

    logger.info(
        "Image shape: (%s, %s, 3), Action dim: %s", img_h, img_w, action_dim
    )
    logger.info("Cameras: %s", list(cam_entries.values()))
    logger.info("State dim: %s", STATE_DIM)
    logger.info("Video codec: %s", args.vcodec)

    features = build_features(action_dim, cam_entries, img_h, img_w)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.output_repo,
        fps=args.fps,
        features=features,
        root=str(output_dir),
        use_videos=True,
        vcodec=args.vcodec,
    )

    total_frames = 0
    failed = 0
    for idx, ep_dir in enumerate(episodes):
        logger.info("[%s/%s] Converting %s", idx + 1, len(episodes), ep_dir)
        try:
            total_frames += convert_episode(
                dataset,
                ep_dir,
                idx,
                camera_names,
                expected_action_dim=action_dim,
            )
        except Exception:
            logger.error("Episode failed, skipping path=%s", ep_dir)
            logger.exception(
                "convert_episode error (continuing with next episode)"
            )
            failed += 1

    if total_frames == 0:
        logger.error("No frames were written; all episodes failed.")
        sys.exit(1)

    dataset.finalize()

    try:
        post_conversion_check(
            repo_id=args.output_repo,
            root=output_dir,
            required_feature_keys=set(features.keys()),
        )
    except Exception as e:
        logger.error("Post-check failed: %s", e)
        sys.exit(1)

    logger.info(
        "Done. %s/%s episodes OK, %s failed, %s total frames -> %s",
        len(episodes) - failed,
        len(episodes),
        failed,
        total_frames,
        output_dir,
    )


if __name__ == "__main__":
    main()
