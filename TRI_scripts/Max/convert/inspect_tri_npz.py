#!/usr/bin/env python3
"""
轉 LeRobot 前，快速檢視某個 episode 的 observations.npz / actions.npz。

用法:
  python inspect_tri_npz.py /path/to/episode_k/processed
  python inspect_tri_npz.py /path/to/observations.npz  # 僅觀測檔（需同目錄有 actions 則一併列）

依賴: numpy（無需 pydrake）
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np


def _npz_keys_and_shapes(npz_path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    out: list[tuple[str, tuple[int, ...], str]] = []
    with zipfile.ZipFile(npz_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".npy"):
                continue
            key = name[: -len(".npy")]
            with zf.open(name) as f:
                ver = np.lib.format.read_magic(f)
                if ver[0] == 1:
                    shape, _, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            out.append((key, shape, str(dtype)))
    return out


def _load_npz_full(npz_path: Path) -> np.lib.npyio.NpzFile:
    return np.load(str(npz_path), allow_pickle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TRI observations.npz / actions.npz before convert.")
    parser.add_argument(
        "path",
        type=Path,
        help=".../processed 目錄，或 observations.npz 檔案路徑",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="印出該幀的 actions 列與（可選）部分 obs 純量",
    )
    args = parser.parse_args()
    p = args.path.resolve()

    if p.is_dir():
        proc = p
        obs_path = proc / "observations.npz"
        act_path = proc / "actions.npz"
    elif p.name == "observations.npz":
        obs_path = p
        proc = p.parent
        act_path = proc / "actions.npz"
    else:
        raise SystemExit("請傳入 processed 目錄或 observations.npz 路徑")

    print("=" * 72)
    print(f"目錄: {proc}")
    print("=" * 72)

    if not obs_path.is_file():
        raise SystemExit(f"找不到 {obs_path}")

    print("\n[observations.npz] 鍵名、shape、dtype（依鍵名字典序）")
    print("-" * 72)
    for key, shape, dtype in _npz_keys_and_shapes(obs_path):
        print(f"  {key}")
        print(f"      shape={shape}  dtype={dtype}")

    if act_path.is_file():
        print("\n[actions.npz]")
        print("-" * 72)
        act = _load_npz_full(act_path)
        print(f"  keys: {list(act.keys())}")
        if "actions" in act:
            a = act["actions"]
            print(f"  actions.shape = {a.shape}  dtype = {a.dtype}")
            t = args.frame
            if a.ndim == 2 and 0 <= t < a.shape[0]:
                row = a[t]
                print(f"\n  actions[{t}] (20,) =")
                print(f"    {row}")
            elif a.ndim == 2:
                print(f"  --frame {t} 超出範圍 T={a.shape[0]}")
    else:
        print(f"\n(無 {act_path.name})")

    # 幾個常用 robot 鍵的第一列摘要
    print("\n[observations 節選 — 與 convert EGO_STATE_KEYS 相關的鍵，第 0 列]")
    print("-" * 72)
    ego_keys = [
        "robot__actual__poses__right::panda__xyz",
        "robot__actual__poses__right::panda__rot_6d",
        "robot__actual__grippers__right::panda_hand",
        "robot__actual__poses__left::panda__xyz",
        "robot__actual__poses__left::panda__rot_6d",
        "robot__actual__grippers__left::panda_hand",
    ]
    obs = _load_npz_full(obs_path)
    t = args.frame
    for k in ego_keys:
        if k not in obs:
            print(f"  (缺鍵) {k}")
            continue
        arr = obs[k]
        if arr.ndim >= 1 and arr.shape[0] > t:
            print(f"  {k}")
            print(f"      row[{t}] = {np.asarray(arr[t]).reshape(-1)}")
        else:
            print(f"  {k} shape={arr.shape} 無法取 frame {t}")


if __name__ == "__main__":
    main()
