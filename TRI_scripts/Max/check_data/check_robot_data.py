"""Peek at npz array shapes via zipfile + numpy header parsing (no full load)."""
import zipfile
import struct
import numpy as np
import yaml
import os
import io

def npz_shapes(path):
    """Return dict of {array_name: (shape, dtype)} without loading full arrays."""
    result = {}
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            key = name[:-4]  # strip .npy
            with zf.open(name) as f:
                version = np.lib.format.read_magic(f)
                if version[0] == 1:
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                result[key] = (shape, dtype)
    return result


ep = "/data/maxshen/Video_data/LBM_sim_egocentric/BimanualPlaceAppleFromBowlOnCuttingBoard/riverway/sim/bc/teleop/2025-01-06T14-24-17-05-00/diffusion_spartan/episode_0/processed"

print("=== observations.npz ===")
obs_shapes = npz_shapes(os.path.join(ep, "observations.npz"))
for k, (shape, dtype) in sorted(obs_shapes.items()):
    print(f"  {k}: shape={shape}, dtype={dtype}")

print("\n=== actions.npz ===")
act_shapes = npz_shapes(os.path.join(ep, "actions.npz"))
for k, (shape, dtype) in sorted(act_shapes.items()):
    print(f"  {k}: shape={shape}, dtype={dtype}")

print("\n=== metadata.yaml ===")
meta = yaml.safe_load(open(os.path.join(ep, "metadata.yaml")))
for k, v in meta.get("camera_id_to_semantic_name", {}).items():
    print(f"  {k} -> {v}")

print("\n=== T consistency check ===")
ref_key = "robot__actual__poses__right::panda__xyz"
T = obs_shapes[ref_key][0][0]
print(f"Reference T={T} (from {ref_key})")
for k, (shape, dtype) in obs_shapes.items():
    if shape[0] != T:
        print(f"  MISMATCH: {k} has T={shape[0]}")
for k, (shape, dtype) in act_shapes.items():
    if shape[0] != T:
        print(f"  MISMATCH (actions): {k} has T={shape[0]}")
print("All arrays match T." if True else "")
