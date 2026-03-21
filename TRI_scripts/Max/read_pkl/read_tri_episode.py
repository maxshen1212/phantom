import pickle
import numpy as np
import gzip
import sys
from collections import Counter

# Add TRI_scripts to Python path
sys.path.append("/data/maxshen/TRI_scripts")

# Read TRI episode.pkl (gzip compressed)
pkl_path = "/data/maxshen/Video_data/LBM_human_egocentric/egoPutKiwiInCenterOfTable/2025-11-13_12-46-27/episode_4 (success)/episode.pkl"

print("=" * 80)
print(f"Reading file: {pkl_path}")
print("=" * 80)

# Use gzip to read compressed pickle file
with gzip.open(pkl_path, "rb") as f:
    data = pickle.load(f)

# Display basic information
print(f"\nData type: {type(data)}")

# If it's a dictionary, display all keys
if isinstance(data, dict):
    print(f"\nContained keys: {list(data.keys())}")
    print(f"Number of dictionary keys: {len(data)}")

    print(f"\n" + "=" * 80)
    print("Detailed information for each key:")
    print("=" * 80)

    for key, value in data.items():
        print(f"\nKey name: '{key}'")
        print(f"  Type: {type(value)}")

        if isinstance(value, (list, tuple)):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                if isinstance(value[0], np.ndarray):
                    print(
                        f"  First element shape: {value[0].shape}, dtype: {value[0].dtype}"
                    )
                    if len(value) > 1:
                        print(
                            f"  All element shapes: {[v.shape if isinstance(v, np.ndarray) else type(v) for v in value[:5]]}"
                        )
                elif isinstance(value[0], dict):
                    print(f"  First element keys: {list(value[0].keys())}")
                elif isinstance(value[0], (int, float, str, bool)):
                    print(f"  First 5 values: {value[:5]}")
                else:
                    print(f"  First element: {value[0]}")

        elif isinstance(value, np.ndarray):
            print(f"  shape: {value.shape}")
            print(f"  dtype: {value.dtype}")
            if value.size <= 10:
                print(f"  值: {value}")
            else:
                print(f"  前幾個值: {value.flatten()[:10]}")

        elif isinstance(value, dict):
            print(f"  Contained keys: {list(value.keys())}")
            for sub_key, sub_value in list(value.items())[
                :3
            ]:  # Display first 3 sub-keys
                print(f"    {sub_key}: {type(sub_value)}", end="")
                if isinstance(sub_value, np.ndarray):
                    print(f" shape={sub_value.shape}")
                elif isinstance(sub_value, (list, tuple)):
                    print(f" len={len(sub_value)}")
                else:
                    print()

        elif isinstance(value, (int, float, str, bool)):
            print(f"  Value: {value}")

        else:
            print(f"  Value: {str(value)[:200]}")

    # Display detailed content of key data
    print(f"\n" + "=" * 80)
    print("Key data content preview:")
    print("=" * 80)

    # Display camera_intrinsics example
    if "camera_intrinsics" in data:
        print(f"\ncamera_intrinsics (first frame):")
        print(data["camera_intrinsics"][0])

    # Display camera_extrinsics example
    if "camera_extrinsics" in data:
        print(f"\ncamera_extrinsics (first frame):")
        print(data["camera_extrinsics"][0])
        # all_same = True

        # for i in range(1, len(data["camera_extrinsics"])):
        #     if not np.allclose(data["camera_extrinsics"][0], data["camera_extrinsics"][i]):
        #         print(f"Difference found at index {i}")
        #         all_same = False
        #         break

        # if all_same:
        #     print("All extrinsics are identical.")

    # Display detailed content of pose_snapshots
    if "pose_snapshots" in data:
        print(f"\npose_snapshots structure (first frame):")
        first_pose = data["pose_snapshots"][0]
        print(f"  Type: {type(first_pose)}")
        print(f"  Keys: {list(first_pose.keys())}")

        for hand_key in ["left", "right"]:
            if hand_key in first_pose:
                hand_data = first_pose[hand_key]
                print(f"\n  '{hand_key}' hand data:")
                print(f"    Type: {type(hand_data)}")
                if isinstance(hand_data, dict):
                    print(f"    Contained keys: {list(hand_data.keys())}")
                    for k, v in hand_data.items():
                        if isinstance(v, np.ndarray):
                            print(
                                f"      {k}: shape={v.shape}, dtype={v.dtype}"
                            )
                            if v.size <= 20:
                                print(f"        Value: {v}")
                        else:
                            print(f"      {k}: {type(v)} = {v}")
                elif hand_data is not None:
                    print(f"    Value: {hand_data}")
                else:
                    print(f"    Value: None")

        if "ts" in first_pose:
            print(f"\n  timestamp: {first_pose['ts']}")

    # Statistics
    print(f"\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    print(f"Total frames: {len(data.get('frame_timestamps', []))}")
    print(f"Task success: {data.get('success', 'N/A')}")

    if "frame_timestamps" in data and len(data["frame_timestamps"]) > 1:
        timestamps = data["frame_timestamps"]
        durations = np.diff(timestamps)
        print(f"Timestamp range: {timestamps[0]:.3f} ~ {timestamps[-1]:.3f}")
        print(f"Total duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
        print(
            f"Average frame interval: {np.mean(durations):.4f} seconds (approx {1/np.mean(durations):.1f} FPS)"
        )

    # Check completeness of hand data
    if "pose_snapshots" in data:
        left_hand_count = sum(
            1 for p in data["pose_snapshots"] if p.get("left") is not None
        )
        right_hand_count = sum(
            1 for p in data["pose_snapshots"] if p.get("right") is not None
        )
        print(f"\nHand data:")
        print(
            f"  Frames with left hand data: {left_hand_count}/{len(data['pose_snapshots'])}"
        )
        print(
            f"  Frames with right hand data: {right_hand_count}/{len(data['pose_snapshots'])}"
        )

elif isinstance(data, (list, tuple)):
    print(f"\nList length: {len(data)}")
    if len(data) > 0:
        print(f"First element type: {type(data[0])}")
        if isinstance(data[0], dict):
            print(f"First element keys: {list(data[0].keys())}")

print("\n" + "=" * 80)
