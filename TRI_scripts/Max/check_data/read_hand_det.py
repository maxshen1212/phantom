import pickle
import numpy as np

# Read hand_det.pkl
pkl_path = "/data/maxshen/test_data/epic/0/hand_det.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# Display basic information
print("=" * 80)
print(f"Reading file: {pkl_path}")
print("=" * 80)
print(f"\nData type: {type(data)}")

if isinstance(data, dict):
    print(f"Number of dictionary keys: {len(data)}")
    print(f"Key range: {min(data.keys())} ~ {max(data.keys())}")

    # View details of the first key
    first_key = list(data.keys())[0]
    first_value = data[first_key]

    print(f"\n" + "=" * 80)
    print(f"Example: Hand detection data for frame {first_key}")
    print("=" * 80)
    print(f"Number of detections per frame: {len(first_value)}")

    # Display detailed information for each hand detection
    for i, hand_det in enumerate(first_value):
        print(f"\n--- Hand Detection {i+1} ---")
        print(f"Type: {type(hand_det)}")

        # Display HandDetection object attributes
        if hasattr(hand_det, "__dict__"):
            for attr, value in hand_det.__dict__.items():
                print(f"  {attr}: {value}")
        else:
            # Try to access common attributes
            try:
                print(f"  bbox: {hand_det.bbox}")
                print(f"    - left: {hand_det.bbox.left:.4f}")
                print(f"    - top: {hand_det.bbox.top:.4f}")
                print(f"    - right: {hand_det.bbox.right:.4f}")
                print(f"    - bottom: {hand_det.bbox.bottom:.4f}")
                print(f"  score: {hand_det.score:.4f}")
                print(f"  state: {hand_det.state} ({hand_det.state.name})")
                print(f"  side: {hand_det.side} ({hand_det.side.name})")
                print(f"  object_offset: {hand_det.object_offset}")
                print(f"    - x: {hand_det.object_offset.x:.4f}")
                print(f"    - y: {hand_det.object_offset.y:.4f}")
            except Exception as e:
                print(f"  Unable to read attributes: {e}")

    # Display statistics
    print(f"\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)

    # Calculate detection count per frame
    detection_counts = [len(v) for v in data.values()]
    print(f"Total frames: {len(data)}")
    print(
        f"Detection count per frame (min/max/avg): {min(detection_counts)}/{max(detection_counts)}/{sum(detection_counts)/len(detection_counts):.2f}"
    )

    # Collect all hand states
    all_states = []
    all_sides = []
    for frame_dets in data.values():
        for det in frame_dets:
            all_states.append(det.state.name)
            all_sides.append(det.side.name)

    from collections import Counter

    print(f"\nHand state distribution:")
    for state, count in Counter(all_states).items():
        print(f"  {state}: {count}")

    print(f"\nHand side distribution:")
    for side, count in Counter(all_sides).items():
        print(f"  {side}: {count}")
