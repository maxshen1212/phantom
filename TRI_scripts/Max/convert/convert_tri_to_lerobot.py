# -*- coding: utf-8 -*-
"""Convert TRI LBM_sim_egocentric data to LeRobot v3 dataset for Diffusion Policy.

================================================================================
【總覽:這支腳本實際做了什麼】
================================================================================
本檔案是格式轉換器, 不是幾何解算器:

  * 不做 task frame <-> 相機 frame 的剛體轉換; 那一步若在 TRI 管線裡做過,
    結果已經寫進 observations.npz 的 __xyz / __rot_6d 欄位。
  * 只做從 npz 逐列讀出 已存在的 float 陣列 -> 依固定順序 concat 成
    LeRobot 的 observation.state; 影像做逐幀切片; action 做原樣拷貝。

因此「ego」的數學定義以產生 npz 的 TRI 為準;
訓練時請讓 lbm_eval 端 lerobot_policy_server 用相同語意還原觀測 observation。

================================================================================
【輸入(每個 episode)】
================================================================================
目錄結構 (典型)::

    <task>/.../diffusion_spartan/episode_k/processed/
        observations.npz   # 含機器人狀態、多相機 RGB 序列 (zip 內多個 .npy)
        actions.npz         # 含 actions 等
        metadata.yaml       # camera_id_to_semantic_name、skills 等

observations.npz 內與本腳本相關的陣列 (鍵名為扁平化後的字串):

    robot__actual__poses__right::panda__xyz      shape (T, 3)
    robot__actual__poses__right::panda__rot_6d   shape (T, 6)
    robot__actual__grippers__right::panda_hand  shape (T, 1)
    robot__actual__poses__left::panda__xyz      shape (T, 3)
    robot__actual__poses__left::panda__rot_6d   shape (T, 6)
    robot__actual__grippers__left::panda_hand   shape (T, 1)
    <camera_hardware_id>                         shape (T, H, W, 3)  uint8 RGB

其中 <camera_hardware_id> 透過 metadata.yaml 的
camera_id_to_semantic_name 對應到語意名 (e.g. scene_right_0)。

================================================================================
【輸出 (LeRobot v3 dataset)】
================================================================================
寫入 --output_dir, 由 LeRobotDataset.create / add_frame / finalize
建立標準 LeRobot 結構 (含 meta/、episodes、可選影片軌等)。

每個時間步 t 寫入一個 frame 字典::

    "task": str
    "observation.state": np.ndarray shape (20,), float32
    "action":            np.ndarray shape (action_dim,), float32  # 通常 20, 與 npz 一致
    "observation.images.<semantic_name>": np.ndarray (H, W, 3) uint8

================================================================================
【20 維 observation.state 的拼接順序 (與本檔 EGO_STATE_KEYS 一致)】
================================================================================
對每個時間步 t, 僅做 concat, 無其他公式:

    s[t] = [  xyz_r[t] (3)  ;  rot6d_r[t] (6)  ;  g_r[t] (1) ;
              xyz_l[t] (3)  ;  rot6d_l[t] (6)  ;  g_l[t] (1)  ]

順序為 **右臂 → 左臂** (與鍵名順序相同)。

================================================================================
【rot_6d 的約定 (本腳本不重新計算, 只轉存)】
================================================================================
npz 內的 6 個數須與 TRI / diffusion_policy 管線一致, 對應
robot_gym.multiarm_spaces_conversions:

  * 設 R ∈ R^{3x3} 為合法旋轉矩陣 (Drake / 列向量為 body 軸在父座標之表達時
    的慣例與匯出端一致)。
  * 編碼 (matrix → 6D):

        rot_6d = [ R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2] ]
               = flatten( R[0:2, :] )   # 前 **兩列 row**, 共 6 個元素

  * 解碼 (6D → R) (Gram-Schmidt 正交化, 見 rotation_6d_to_matrix):
        a1 = d[0:3], a2 = d[3:6]; 正規化 a1; 將 a2 對 a1 去分量後正規化;
        a3 = a1 x a2; 堆疊成 3x3 旋轉矩陣 (列為正交基)。

**注意**:此與部分論文 / LeRobot XvLA 使用的「取 R 的兩個 column」6D 不同, 若混用會導致旋轉語意錯亂。

================================================================================
【xyz 的約定】
================================================================================
本腳本不變換座標。npz 內 __xyz 一般表示 末端在「主視角相機座標系」下的 平移 (ego-centric position),
與上述 rot_6d 同一參考座標系; 精確定義以 TRI 錄製匯出為準。

================================================================================
【action】
================================================================================
actions.npz 中鍵 "actions" 的列向量 照原樣 astype(np.float32) 寫入frame["action"]。
常見情況為與 observation.state 同維 (20)、同順序,
且為絕對 ego 目標 (仍須以資料集實際內容為準)。

================================================================================
【效能說明】
================================================================================
By default we load only the main egocentric scene camera: scene_right_0
(Kyle's ego-centric view). Masquerade-style pretraining also emphasizes this
views.
Add more views with --cameras ... or use --all-cameras when you need the full set.

Usage:
    # Full dataset:
    python convert_tri_to_lerobot.py \
        --input_dir /data/maxshen/Video_data/LBM_sim_egocentric/BimanualPlaceAppleFromBowlOnCuttingBoard \
        --output_repo lbm_sim/ego_BimanualPlaceAppleFromBowlOnCuttingBoard \
        --output_dir /data/maxshen/lerobot_training_data/ego_BimanualPlaceAppleFromBowlOnCuttingBoard \
        --fps 10

    # Quick test (1 episode):
    python convert_tri_to_lerobot.py \
        --input_dir /data/maxshen/Video_data/LBM_sim_egocentric/BimanualPlaceAppleFromBowlOnCuttingBoard \
        --output_repo lbm_sim/ego_BimanualPlaceAppleFromBowlOnCuttingBoard \
        --output_dir /data/maxshen/lerobot_training_data/ego_BimanualPlaceAppleFromBowlOnCuttingBoard \
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

# 預設只載入主第一人稱場景相機; 語意名須出現在該 episode 的 metadata.yaml
DEFAULT_CAMERAS: tuple[str, ...] = ("scene_right_0",)

# 與 LeRobot observation.state 欄位順序對應; 每項為 (npz 內陣列鍵名, 該段長度)。
# 拼接公式:state_row = np.concatenate([a0, a1, ... , ai], axis=0), 其中 ai 為該列 t 的子向量。
EGO_STATE_KEYS: list[tuple[str, int]] = [
    ("robot__actual__poses__right::panda__xyz", 3),
    ("robot__actual__poses__right::panda__rot_6d", 6),
    ("robot__actual__grippers__right::panda_hand", 1),
    ("robot__actual__poses__left::panda__xyz", 3),
    ("robot__actual__poses__left::panda__rot_6d", 6),
    ("robot__actual__grippers__left::panda_hand", 1),
]
STATE_DIM = sum(d for _, d in EGO_STATE_KEYS)  # 3+6+1+3+6+1 = 20
ACTIONS_KEY = "actions"


def npz_array_shape(path: Path, key: str) -> tuple:
    """讀 zip-npz 內某個陣列的 header, 取得 shape (不載入完整資料)。"""
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(key + ".npy") as f:
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
    return shape


def discover_episodes(input_dir: Path) -> list[Path]:
    """掃描 input_dir 下所有 ``.../processed/observations.npz``, 回傳 processed 目錄路徑列表。"""
    episodes = sorted(
        p.parent
        for p in input_dir.rglob("observations.npz")
        if p.parent.name == "processed"
    )
    logger.info(f"Discovered {len(episodes)} episodes under {input_dir}")
    return episodes


def task_name_from_episode(processed_dir: Path) -> str:
    """從 metadata.yaml 的 skills 取任務名, 否則從路徑推斷。"""
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
    """
    輸入:某 episode 的 processed 目錄、欲載入的語意相機名列表 (None = 全部)。
    輸出:dict hardware_id -> semantic_name, 提供從 observations.npz 取 RGB 陣列鍵。

    轉換:僅查表 metadata.yaml 的 camera_id_to_semantic_name。
    """
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
    """
    單一 episode 轉換。

    輸入檔:
        processed_dir/observations.npz, processed_dir/actions.npz, metadata.yaml (經 resolve_camera_entries)

    對每個時間步 t (共 T 步):
        * observation.state:
              state_all[t] = concat( EGO_STATE_KEYS 各欄第 t row.astype(float32) )
              維度 (20,)。
        * action:
              actions[t].astype(float32), 維度 (action_dim,)
        * observation.images.<sem>:
              cam_arrays[sem][t], 維度 (H,W,3), uint8; 公式:I_t = stack[t] (單純索引)

    回傳: 本 episode 的幀數 T。
    """
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

    # 沿特徵維拼接: state_all[t] = [right block || left block]
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
    """
    建立 LeRobotDataset.create 所需的 features 描述 dict。

    observation.state 的 names.axes 僅供人類閱讀 metadata; 張量實際順序必須與
    EGO_STATE_KEYS (右→左)一致。
    """
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {
                "axes": [
                    # 順序須與 EGO_STATE_KEYS 一致:右腕 → 左腕
                    "ego_right_xyz_x", "ego_right_xyz_y", "ego_right_xyz_z",
                    "ego_right_rot6d_0", "ego_right_rot6d_1", "ego_right_rot6d_2",
                    "ego_right_rot6d_3", "ego_right_rot6d_4", "ego_right_rot6d_5",
                    "ego_right_gripper",
                    "ego_left_xyz_x", "ego_left_xyz_y", "ego_left_xyz_z",
                    "ego_left_rot6d_0", "ego_left_rot6d_1", "ego_left_rot6d_2",
                    "ego_left_rot6d_3", "ego_left_rot6d_4", "ego_left_rot6d_5",
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
    logger.info(f"State dim: {STATE_DIM} (bimanual ego proprioception, right then left)")
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
