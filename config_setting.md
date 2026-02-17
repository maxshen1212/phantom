# Config path:
  /phantom/configs
# Camera path:
  /phantom/phantom/camera
# Output path:
  /phantom/data/processed
# Input path:
  /phantom/data/raw

# Input data structure:
  /phantom/data/raw/[demo_name]/[index]/hand_det.pkl video_L.mp4

# config format:
```yaml
# Default configuration (PHANTOM paper settings)
debug: false
verbose: false
skip_existing: false
n_processes: 1
data_root_dir: "../data/raw_data/"
processed_data_root_dir: "../data/processed_data/"
demo_name: ""

# Processing settings
mode: ["bbox"]  # Default processing mode
demo_num: null  # Process specific demo number (null = process all videos in the root folder)

# Additional settings
debug_cameras: [] # Add other robomimic cameras like sideview, etc. Warning: this significantly slows down the processing time


# EPIC-KITCHENS configuration override
input_resolution: 256
output_resolution: 256
robot: "Kinova3"
gripper: "Robotiq85"
square: false
epic: true
bimanual_setup: "shoulders"
target_hand: "both"
constrained_hand: false
depth_for_overlay: false
render: false
camera_intrinsics: "camera/camera_intrinsics_epic.json"
camera_extrinsics: "camera/camera_extrinsics_ego_bimanual_shoulders.json"

```

# camer intrinsics format:
```json
{
    "left": {
        "fx": 1057.7322998046875,
        "fy": 1057.7322998046875,
        "cx": 972.5150756835938,
        "cy": 552.568359375,
        "disto": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "v_fov": 54.09259796142578,
        "h_fov": 84.45639038085938,
        "d_fov": 92.32276916503906
    },
    "right": {
        "fx": 1057.7322998046875,
        "fy": 1057.7322998046875,
        "cx": 972.5150756835938,
        "cy": 552.568359375,
        "disto": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "v_fov": 54.09259796142578,
        "h_fov": 84.45639038085938,
        "d_fov": 92.32276916503906
    }
}
```

# camer extrinsics format:
```json
[
    {
        "camera_base_ori": [
            [
                0.9842690634302423,
                -0.053375086066005106,
                0.1684206369825258
            ],
            [
                -0.1763762231197722,
                -0.35235905397979306,
                0.9190944048336218
            ],
            [
                0.010287793357058851,
                -0.934341584895969,
                -0.3562302121408726
            ]
        ],
        "camera_base_ori_rotvec": [
            -1.930138005212092,
            0.16467696378244215,
            -0.12809137765065973
        ],
        "camera_base_pos": [
            0.3407932803063093,
            -0.40868423448040403,
            0.39911982578151795
        ],
        "camera_base_quat": [
            0.8204965462375373,
            -0.07000374049084156,
            0.054451304871138306,
            -0.564729979129313
        ],
        "p_marker_ee": [
            -0.01874144739551215,
            0.029611448317719172,
            -0.013687685723932594
        ]
    }
]
```

# hand_det.pkl format
```python
"""Minimal epic_kitchens.hoa.types for pickle compatibility"""

from enum import Enum, unique
from dataclasses import dataclass
import numpy as np


@unique
class HandSide(Enum):
    LEFT = 0
    RIGHT = 1


@unique
class HandState(Enum):
    NO_CONTACT = 0
    SELF_CONTACT = 1
    ANOTHER_PERSON = 2
    PORTABLE_OBJECT = 3
    STATIONARY_OBJECT = 4


@dataclass
class FloatVector:
    x: np.float32
    y: np.float32


@dataclass
class BBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass
class HandDetection:
    bbox: BBox
    score: np.float32
    state: HandState
    side: HandSide
    object_offset: FloatVector
```
