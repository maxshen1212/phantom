# Ego-Centric 20-Dimensional State and Action Layout

This note specifies the naming, dimensionality, and **index ordering** of the 20-dimensional proprioceptive state and action vectors used when converting TRI `npz` archives to LeRobot format and when serving policies over gRPC.

---

## Executive summary

The 20-dimensional `observation.state` vector and the 20-dimensional `action` vector **do not share the same per-index semantics**:

| Quantity | Ordering convention | Structure |
|----------|---------------------|-----------|
| `observation.state` | Interleaved per arm | For each arm: translation (3) → rotation-6D (6) → gripper (1); right arm first, then left. |
| `action` | TRI documentation order | All right-arm pose components, then all left-arm pose components, then right gripper, then left gripper. |

The conversion pipeline and `lerobot_policy_server` are aligned with this distinction. **Do not** apply the same slice definitions to `observation.state` and to `action` without an explicit permutation or documentation update.

---

## 1. `observation.state` (20D, interleaved)

**Sources and implementation**

- **TRI archive:** Multiple keys in `observations.npz` are concatenated along the feature axis according to `EGO_STATE_KEYS` in [`convert_tri_to_lerobot.py`](convert_tri_to_lerobot.py).
- **Inference:** The same ordering is produced by `_proprio_20d_numpy_from_observation` in [`lerobot_policy_server.py`](/data/maxshen/lbm_eval/grpc_workspace/lerobot_policy_server.py).

**Index map** (slice notation is half-open: `[start:end)` except single indices):

| Component | Index range | Dimension |
|-----------|-------------|-----------|
| Right end-effector position (xyz) | `[0, 3)` | 3 |
| Right end-effector rotation (6D) | `[3, 9)` | 6 |
| Right gripper | `9` | 1 |
| Left end-effector position (xyz) | `[10, 13)` | 3 |
| Left end-effector rotation (6D) | `[13, 19)` | 6 |
| Left gripper | `19` | 1 |

**Geometric conventions**

- End-effector poses are expressed in the **primary camera (ego) frame**.
- The 6D rotation follows the TRI convention (two columns of the rotation matrix flattened), consistent with `robot_gym.matrix_to_rotation_6d`.

---

## 2. `action` (20D, TRI specification order)

**Sources and implementation**

- **TRI archive:** Array `actions` in `actions.npz`, shape `(T, 20)`.
- **Specification:** See *The actions archive* in [`TRAINING_DATA_FORMAT.md`](/data/maxshen/lbm_eval/TRAINING_DATA_FORMAT.md).
- **Conversion:** Rows are written **without reordering** into the LeRobot `action` field.

**Index map**

| Component | Index range | Dimension |
|-----------|-------------|-----------|
| Right translation (m) | `[0, 3)` | 3 |
| Right rotation (6D) | `[3, 9)` | 6 |
| Left translation (m) | `[9, 12)` | 3 |
| Left rotation (6D) | `[12, 18)` | 6 |
| Right gripper (RG) | `18` | 1 |
| Left gripper (LG) | `19` | 1 |

**Inference**

- The helper `_action_tensor_to_poses_and_grippers` slices the model output according to the table above, then transforms poses into the task frame via the primary-camera extrinsic `X_TC` to populate `PosesAndGrippers`.

---

## 3. Images and cameras

- **Default semantic camera name:** `scene_right_0`. This name must agree across the dataset (`observation.images.*`), conversion metadata, and server configuration.
- **Server tensor layout:** RGB is converted from `HWC` `uint8` to `CHW` `float32` scaled by `1/255`, with batch dimension `(1, 3, H, W)`.

---

## 4. Compatibility when changing layouts

If either of the following is modified, the other components of the stack must be updated accordingly:

- Reordering `action` during conversion (e.g., to match the interleaved state layout), or
- Changing the concatenation order for `observation.state`.

**Required follow-up steps**

1. Update slicing and/or concatenation logic in `lerobot_policy_server` to match the new layout.
2. Retrain policies; checkpoints trained under a previous layout must not be mixed with the new tensor convention without an explicit adapter.

---

## 5. Reference locations

| Topic | Path |
|-------|------|
| State key order (`EGO_STATE_KEYS`) | `phantom/TRI_scripts/Max/convert/convert_tri_to_lerobot.py` |
| gRPC observation assembly and action decoding | `lbm_eval/grpc_workspace/lerobot_policy_server.py` |

For offline `npz` parity checks, refer to the module docstring and notes at the top of `lerobot_policy_server.py`.
