"""
TRI to EPIC Hand Detection Converter

=============================================================================
PURPOSE
=============================================================================
Converts TRI episode.pkl (3D hand skeletons) to EPIC-KITCHENS hand_det.pkl (2D bounding boxes).
Pipeline: Load => Extract 3D joints => Project to 2D => Compute bbox => Save EPIC format

=============================================================================
COORDINATE SYSTEM TRANSFORMATIONS
=============================================================================

This converter performs a chain of coordinate transformations:

1. WORLD FRAME (ARKit coordinate system)
   - Origin: Defined by ARKit SLAM system
   - Units: Meters
   - Data: Hand joint 3D positions from Vision Pro tracking
   - Source: episode.pkl['pose_snapshots'][i]['left/right']['response']

2. DEVICE FRAME (Vision Pro headset)
   - Transform: world_T_device (4x4 SE(3) matrix)
   - Origin: Center of Vision Pro headset
   - Contains: Device position and orientation in world
   - Source: skeleton_response.device (translation + quaternion rotation)

3. CAMERA FRAME (Vision Pro main camera)
   - Transform: device_T_camera (computed from camera_extrinsics)
   - Origin: Camera optical center
   - Axes: X=right, Y=down, Z=forward (OpenCV convention)
   - Units: Meters

   Full transformation chain:
   camera_T_world = inv(world_T_device @ device_T_camera)

   Where:
   - world_T_device: from skeleton_response.device
   - camera_extrinsics: stored TRANSPOSED in episode.pkl
   - device_T_camera = inv(camera_extrinsics.T)

4. IMAGE FRAME (2D pixel coordinates)
   - Transform: Pinhole camera projection using intrinsics
   - Origin: Top-left corner of image
   - Axes: u=horizontal (right+), v=vertical (down+)
   - Units: Pixels
   - Range: u ∈ [0, 1920], v ∈ [0, 1080] (original resolution)

   Projection equations:
   u = fx * (x_camera / z_camera) + cx
   v = fy * (y_camera / z_camera) + cy

   Where camera_intrinsics K:
   [[fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]]

5. NORMALIZED FRAME (EPIC format)
   - Transform: Divide by image dimensions
   - Range: [0, 1] * [0, 1]
   - Format: BBox(left, top, right, bottom) with all values in [0, 1]

=============================================================================
DATA FLOW
=============================================================================

Input (TRI format):
├── episode.pkl (gzipped)
│   ├── frame_timestamps: List[float]           # Frame times in seconds
│   ├── pose_snapshots: List[Dict]              # Per-frame data
│   │   └── {left/right: {response: skeleton}}  # Hand skeletal data
│   ├── camera_intrinsics: List[ndarray(3,3)]   # K matrices per frame
│   └── camera_extrinsics: List[ndarray(4,4)]   # Camera poses per frame
└── main_camera.mp4 (1920*1080)                 # RGB video

Processing:
└── For each frame:
    ├── Extract device_pose: world_T_device (4*4)
    ├── Extract hand_joints: (26, 3) in world frame
    ├── Project to 2D: (26, 3) => (26, 2) pixel coords
    ├── Compute bbox: min/max of 2D joints + padding
    └── Normalize: pixel coords => [0,1] range

Output (EPIC format):
├── hand_det.pkl
│   └── {frame_id: List[HandDetection]}         # "0", "1", ... => detections
│       └── HandDetection(bbox, score, side, state, object_offset)
├── video_L.mp4 (456*256)                       # Downsampled video
├── camera_intrinsics_tri.json                  # Scaled K matrix
└── camera_extrinsics_tri.json                  # Reused EPIC calibration

=============================================================================
KEY IMPLEMENTATION NOTES
=============================================================================

1. Camera Extrinsics Storage:
   - episode.pkl stores camera_extrinsics TRANSPOSED
   - Must transpose back before inverting: device_T_camera = inv(extrinsics.T)

2. Joint Transform Composition:
   - Hand joints are LOCAL to anchor in ARKit data
   - World position: world_T_anchor @ anchor_T_joint
   - Matrix orders matter: compose right-to-left

3. Depth Handling:
   - No z > 0 check during projection (matches original behavior)
   - Points behind camera (z < 0) produce invalid projections
   - Bbox validation catches out-of-bounds results

4. Video Downsampling:
   - Required: E2FGVI cannot handle 1920*1080 in EPIC mode (GPU OOM)
   - Intrinsics scale: fx_new = fx_old * (456/1920)
   - Method: cv2.INTER_AREA (best for downsampling)

5. Extrinsics Reuse:
   - TRI uses EPIC ego_bimanual_shoulders extrinsics
   - Reason: Same viewing geometry (head-mounted, downward)
   - Purpose: Maps HaMeR output to robot workspace
   - Alternative (identity) places points outside reachable space

=============================================================================
DEPENDENCIES
=============================================================================

- numpy: Numerical operations and array manipulation
- scipy.spatial.transform: Quaternion to rotation matrix conversion
- opencv-python (cv2): Video reading, writing, and resizing
- pickle/gzip: Loading compressed TRI episode data
- epic_kitchens.hoa.types: EPIC format dataclasses (HandDetection, BBox, etc.)

=============================================================================
"""

import pickle
import gzip
import json
import numpy as np
import sys
import os
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation

# ============================================================================
# PATH SETUP: Add TRI_scripts to Python path for protobuf imports
# ============================================================================
# Kyle's directory contains tracker_pb2.py (protobuf definitions for TRI data)
tri_scripts_path = "/data/maxshen/phantom/TRI_scripts/Kyle"
if tri_scripts_path not in sys.path:
    sys.path.insert(0, tri_scripts_path)  # Prepend to ensure priority

# ============================================================================
# EPIC-KITCHENS TYPE IMPORTS
# ============================================================================
# Import EPIC format dataclasses for hand detection output
# - HandDetection: Container for one hand detection (bbox + metadata)
# - BBox: Bounding box with normalized coords (left, top, right, bottom)
# - HandSide: Enum (LEFT, RIGHT)
# - HandState: Enum (NO_CONTACT, CONTACT, PORTABLE_OBJECT, STATIC_OBJECT)
# - FloatVector: 2D vector for object_offset
from epic_kitchens.hoa.types import (
    HandDetection,
    BBox,
    HandSide,
    HandState,
    FloatVector,
)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Original Vision Pro camera resolution (pixels)
# - Vision Pro main camera captures at Full HD resolution
# - These dimensions are used for original bounding box computation
DEFAULT_IMG_WIDTH = 1920  # Width in pixels
DEFAULT_IMG_HEIGHT = 1080  # Height in pixels

# Bounding box padding factor
# - Expands bbox by 10% on each side to avoid clipping hand edges
# - Applied after computing min/max 2D joint coordinates
# - Example: if bbox width is 100px, add 10px padding on left and right
BBOX_PADDING = 0.1  # 10% padding factor

# Target resolution for video downsampling
# - E2FGVI inpainting model cannot handle 1920x1080 in EPIC mode (GPU OOM)
# - EPIC-KITCHENS dataset uses 456x256 resolution
# - Downsampling ratio: 0.2375x (width), 0.2370x (height)
TARGET_WIDTH = 456  # Target width matches EPIC-KITCHENS standard
TARGET_HEIGHT = 256  # Target height matches EPIC-KITCHENS standard


def parse_to_se3(transform) -> np.ndarray:
    """
    Convert protobuf transform to 4x4 SE(3) transformation matrix.

    SE(3) represents rigid body transformations (rotation + translation) in 3D space.
    The resulting matrix can be used to transform points from one coordinate frame to another.

    Args:
        transform: Protobuf transform object with translation (x,y,z) and rotation (quaternion)

    Returns:
        4x4 numpy array representing the SE(3) transformation matrix:
        [[R R R tx]
         [R R R ty]
         [R R R tz]
         [0 0 0  1]]
        where R is the 3x3 rotation matrix and t is the translation vector
    """
    # Extract translation vector (3D position in meters)
    # - Protobuf stores translation as separate x, y, z fields
    # - Create numpy array for matrix operations
    t = np.array(
        [
            transform.translation.x,  # X component (meters)
            transform.translation.y,  # Y component (meters)
            transform.translation.z,  # Z component (meters)
        ]
    )

    # Extract rotation as quaternion (x, y, z, w)
    # - Protobuf stores rotation as quaternion with 4 components
    # - Quaternion format: [qx, qy, qz, qw] (scalar-last convention)
    # - Represents 3D rotation without gimbal lock
    q = np.array(
        [
            transform.rotation.x,  # Quaternion X (imaginary part i)
            transform.rotation.y,  # Quaternion Y (imaginary part j)
            transform.rotation.z,  # Quaternion Z (imaginary part k)
            transform.rotation.w,  # Quaternion W (real part)
        ]
    )

    # Build 4x4 SE(3) transformation matrix
    # Format: [[R | t],  where R is 3x3 rotation, t is 3x1 translation
    #          [0 | 1]]        0 is 1x3 zeros,    1 is scalar
    se3 = np.eye(4)  # Initialize as 4x4 identity matrix

    # Convert quaternion to 3x3 rotation matrix and insert into top-left
    # - scipy.spatial.transform.Rotation handles quaternion to matrix conversion
    # - Uses standard right-handed coordinate system
    se3[:3, :3] = Rotation.from_quat(
        q
    ).as_matrix()  # Set rotation part (top-left 3x3)

    # Insert translation vector into rightmost column
    se3[:3, 3] = t  # Set translation part (top-right 3x1)

    # Bottom row [0 0 0 1] is already set by np.eye(4)
    return se3  # Shape: (4, 4), represents frame transformation


def extract_hand_joints_3d(skeleton_response) -> np.ndarray:
    """
    Extract 26 hand joint positions in world coordinates from ARKit hand skeleton.

    ARKit provides hand skeletal data with 25 joints plus 1 anchor point.
    The anchor_transform is already in world coordinates from ARKit.
    Each joint's transform is relative to the anchor, so we compose them
    to get world coordinates: joint_world = anchor_world @ joint_local

    Args:
        skeleton_response: Protobuf skeleton response containing hand data

    Returns:
        (26, 3) numpy array of 3D joint positions in world coordinates
        Order: [anchor, wrist, thumb(4), index(5), middle(5), ring(5), pinky(5)]
    """
    # Validate input has hand skeletal data
    if not hasattr(skeleton_response, "hand"):
        raise ValueError("Skeleton has no 'hand' attribute")

    # Extract ARKit hand skeleton structure from protobuf
    hand_skeleton = skeleton_response.hand.hand_skeleton

    # Initialize list to store 26 joint positions (each joint is 3D point)
    joints_3d = []
    # ARKit hand joint names (25 joints total)
    # - ARKit provides a hierarchical hand skeleton structure
    # - Each finger has 5 joints: metacarpal, knuckle, intermediate_base, intermediate_tip, tip
    # - Thumb has 4 joints (no metacarpal in ARKit's model)
    # - Joint order matters for consistent indexing across frames
    joint_names = [
        "wrist",  # Joint 1: Wrist (base of hand)
        "thumb_knuckle",  # Joint 2: Thumb CMC joint
        "thumb_intermediate_base",  # Joint 3: Thumb MCP joint
        "thumb_intermediate_tip",  # Joint 4: Thumb IP joint
        "thumb_tip",  # Joint 5: Thumb tip
        "index_finger_metacarpal",  # Joint 6: Index finger metacarpal base
        "index_finger_knuckle",  # Joint 7: Index finger MCP joint
        "index_finger_intermediate_base",  # Joint 8: Index finger PIP joint
        "index_finger_intermediate_tip",  # Joint 9: Index finger DIP joint
        "index_finger_tip",  # Joint 10: Index finger tip
        "middle_finger_metacarpal",  # Joint 11: Middle finger metacarpal base
        "middle_finger_knuckle",  # Joint 12: Middle finger MCP joint
        "middle_finger_intermediate_base",  # Joint 13: Middle finger PIP joint
        "middle_finger_intermediate_tip",  # Joint 14: Middle finger DIP joint
        "middle_finger_tip",  # Joint 15: Middle finger tip
        "ring_finger_metacarpal",  # Joint 16: Ring finger metacarpal base
        "ring_finger_knuckle",  # Joint 17: Ring finger MCP joint
        "ring_finger_intermediate_base",  # Joint 18: Ring finger PIP joint
        "ring_finger_intermediate_tip",  # Joint 19: Ring finger DIP joint
        "ring_finger_tip",  # Joint 20: Ring finger tip
        "little_finger_metacarpal",  # Joint 21: Pinky metacarpal base
        "little_finger_knuckle",  # Joint 22: Pinky MCP joint
        "little_finger_intermediate_base",  # Joint 23: Pinky PIP joint
        "little_finger_intermediate_tip",  # Joint 24: Pinky DIP joint
        "little_finger_tip",  # Joint 25: Pinky tip
    ]

    # Get anchor transform (already in world coordinates from ARKit)
    # - ARKit anchor represents the hand's root position in world space
    # - This is the reference point for all hand joints
    # - Transform format: 4x4 SE(3) matrix (world_T_anchor)
    anchor_transform = skeleton_response.hand.anchor_transform
    anchor_se3 = parse_to_se3(
        anchor_transform
    )  # Convert protobuf to numpy matrix

    # Joint 0: Extract anchor position (wrist region) in world coordinates
    # - anchor_se3[:3, 3] extracts the translation vector (rightmost column, top 3 rows)
    # - This gives us the 3D position (x, y, z) in world frame (meters)
    joints_3d.append(
        anchor_se3[:3, 3]
    )  # Shape: (3,) - [x_world, y_world, z_world]

    # Extract remaining 25 joints by composing transforms
    # - Each joint's transform is LOCAL (relative to anchor)
    # - To get WORLD coordinates, we compose: world_T_joint = world_T_anchor @ anchor_T_joint
    # - Matrix multiplication order matters: right-to-left transformation chain
    for joint_name in joint_names:
        if hasattr(hand_skeleton, joint_name):
            # Get joint's local transform (relative to anchor)
            joint = getattr(hand_skeleton, joint_name)

            # Convert joint's local transform to SE(3) matrix
            # Format: anchor_T_joint (4x4 matrix)
            joint_se3 = parse_to_se3(joint)

            # Compose transforms to get world coordinates
            # - world_T_joint = world_T_anchor @ anchor_T_joint
            # - Matrix multiplication: @ operator for numpy arrays
            # - Result: 4x4 matrix representing joint in world frame
            joint_world_se3 = anchor_se3 @ joint_se3

            # Extract 3D position from transformation matrix
            # - joint_world_se3[:3, 3]: rightmost column, top 3 elements
            # - This is the translation part (x, y, z) in world coordinates (meters)
            # - Ignore rotation part (we only need position for bbox computation)
            joints_3d.append(joint_world_se3[:3, 3])  # Shape: (3,) - [x, y, z]

    # Convert list to numpy array for efficient computation
    # - Shape: (26, 3) where 26 = number of joints, 3 = (x, y, z)
    # - Row i contains [x_i, y_i, z_i] for joint i in world coordinates
    # - Units: meters (ARKit world coordinate system)
    return np.array(joints_3d)  # Shape: (26, 3)


def get_device_pose_from_skeleton(skeleton_response) -> np.ndarray:
    """
    Extract Vision Pro device pose in world coordinates.

    The device pose represents where the Vision Pro headset is located and oriented
    in the world coordinate system. This is needed to transform between world
    coordinates (where hand joints are) and camera coordinates (for projection).

    Args:
        skeleton_response: Protobuf skeleton response containing device pose data

    Returns:
        4x4 transformation matrix (world_T_device) representing device pose in world frame
    """
    return parse_to_se3(skeleton_response.device)


def project_3d_to_2d(
    points_3d: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    device_extrinsics: np.ndarray,
    debug: bool = False,
) -> np.ndarray:
    """
    Project 3D world points to 2D image pixel coordinates.

    Transformation pipeline:
    1. World coordinates => Camera coordinates using extrinsics
    2. Camera coordinates => Image coordinates using intrinsics (pinhole model)

    Coordinate frames:
    - World: ARKit world coordinate system (where hand joints are defined)
    - Device: Vision Pro headset pose in world
    - Camera: Vision Pro camera frame (relative to device)
    - Image: 2D pixel coordinates (u, v)

    Args:
        points_3d: (N, 3) array of 3D points in world coordinates
        camera_intrinsics: 3x3 camera intrinsic matrix (focal length, principal point)
        camera_extrinsics: 4x4 camera extrinsic matrix (stored transposed)
        device_extrinsics: 4x4 device pose in world (world_T_device)
        debug: If True, print intermediate transformation results

    Returns:
        (N, 2) array of 2D pixel coordinates (u, v)
    """
    # =========================================================================
    # STEP 1: Compute device_T_camera (transform from camera frame to device frame)
    # =========================================================================
    #
    # Explanation of camera_extrinsics format:
    # - camera_extrinsics is stored as TRANSPOSED in the episode.pkl file
    # - Original meaning: camera_T_device (camera FROM device)
    # - We need the inverse: device_T_camera (device FROM camera)
    #
    # Computation:
    # 1. camera_extrinsics.T: transpose back to get camera_T_device
    # 2. np.linalg.inv(): invert to get device_T_camera
    #
    # Result: device_T_camera transforms points from camera frame to device frame
    # - Format: 4x4 SE(3) matrix
    # - Direction: p_device = device_T_camera @ p_camera
    device_T_camera = np.linalg.inv(camera_extrinsics.T)

    # =========================================================================
    # STEP 2: Compose transformation chain from world to camera
    # =========================================================================
    #
    # Transformation chain: World => Device => Camera
    #
    # Given:
    # - world_T_device (device_extrinsics): transforms device frame to world frame
    # - device_T_camera (computed above): transforms camera frame to device frame
    #
    # Goal: Get camera_T_world to transform world points to camera frame
    #
    # Forward composition:
    #   world_T_camera = world_T_device @ device_T_camera
    #   (read right-to-left: camera => device => world)
    #
    # Then invert to get reverse direction:
    #   camera_T_world = inv(world_T_camera)
    #   (transforms: world => device => camera)
    #
    # Matrix multiplication order explanation:
    # - To go from camera to world: p_world = world_T_device @ (device_T_camera @ p_camera)
    # - Associativity: p_world = (world_T_device @ device_T_camera) @ p_camera
    # - Therefore: world_T_camera = world_T_device @ device_T_camera

    # Compose forward transformation (camera to world)
    world_T_camera = device_extrinsics @ device_T_camera

    # Invert to get backward transformation (world to camera)
    # - This is what we need to transform 3D world points into camera frame
    # - Direction: p_camera = camera_T_world @ p_world
    camera_T_world = np.linalg.inv(world_T_camera)

    # Debug output: print transformation matrices if requested
    if debug:
        print(f"device_extrinsics (world_T_device):\n{device_extrinsics}")
        print(f"camera_extrinsics.T (camera_T_device):\n{camera_extrinsics.T}")
        print(
            f"First point in world coordinates [x, y, z] meters: {points_3d[0]}"
        )

    # =========================================================================
    # STEP 3: Transform world points to camera frame
    # =========================================================================
    #
    # Input: points_3d (N, 3) - 3D points in world coordinates [x_w, y_w, z_w]
    # Output: points_camera (N, 3) - 3D points in camera coordinates [x_c, y_c, z_c]
    #
    # Process:
    # 1. Convert to homogeneous coordinates (add w=1 as 4th dimension)
    #    - Homogeneous form: [x, y, z, 1]
    #    - Allows SE(3) matrix multiplication to handle rotation + translation
    # 2. Apply camera_T_world transformation
    # 3. Convert back to Euclidean coordinates (remove 4th dimension)
    #
    # Camera coordinate system (OpenCV convention):
    # - X axis: right (increases to the right in image)
    # - Y axis: down (increases downward in image)
    # - Z axis: forward (increases away from camera, into the scene)
    # - Origin: camera optical center

    N = points_3d.shape[0]  # Number of points to transform

    # Convert to homogeneous coordinates: append column of ones
    # - points_3d: (N, 3) array of [x, y, z]
    # - np.ones((N, 1)): (N, 1) column of ones
    # - np.hstack: horizontal stack => (N, 4) array of [x, y, z, 1]
    points_world_hom = np.hstack([points_3d, np.ones((N, 1))])  # Shape: (N, 4)

    # Apply transformation using matrix multiplication
    # - camera_T_world: (4, 4) transformation matrix
    # - points_world_hom.T: (4, N) transposed points (each column is a point)
    # - Multiply: (4, 4) @ (4, N) = (4, N) result
    # - Transpose back: (4, N).T = (N, 4)
    # - Extract first 3 columns: [:, :3] = (N, 3) [x_camera, y_camera, z_camera]
    points_camera = (camera_T_world @ points_world_hom.T).T[
        :, :3
    ]  # Shape: (N, 3)

    # Debug output: verify camera frame transformation
    if debug:
        print(
            f"First point in camera coordinates [x_c, y_c, z_c] meters: {points_camera[0]}"
        )
        print(
            f"Camera z values (depth): min={points_camera[:, 2].min():.3f}m, max={points_camera[:, 2].max():.3f}m"
        )
        # Note: Positive z means in front of camera (valid), negative z means behind camera (invalid)

    # =========================================================================
    # STEP 4: Project 3D camera points to 2D image pixels (Pinhole Camera Model)
    # =========================================================================
    #
    # Pinhole camera projection equations:
    #   u = fx * (x_c / z_c) + cx
    #   v = fy * (y_c / z_c) + cy
    #
    # Where:
    # - (x_c, y_c, z_c): 3D point in camera frame (meters)
    # - (u, v): 2D pixel coordinates in image (pixels)
    # - fx, fy: focal lengths in pixels (measure zoom level)
    # - cx, cy: principal point in pixels (image center, usually ~width/2, ~height/2)
    #
    # Intrinsic matrix K format:
    #   K = [[fx,  0, cx],
    #        [ 0, fy, cy],
    #        [ 0,  0,  1]]
    #
    # Perspective division: divide x, y by z (depth)
    # - Points further away (larger z) project to smaller pixel displacement
    # - This creates perspective effect (far objects appear smaller)
    #
    # Note: This implementation does NOT check if z > 0 (depth validity)
    # - Matches original implementation behavior
    # - Points behind camera (z < 0) will produce invalid projections
    # - Consider adding z > 0 check in future for robustness

    # Initialize output array for 2D pixel coordinates
    points_2d = np.zeros((N, 2))  # Shape: (N, 2) for N points, 2 coords (u, v)

    # Project each 3D point individually
    for i in range(N):
        # Extract 3D coordinates in camera frame
        x, y, z = points_camera[i]  # x_c, y_c, z_c in meters

        # Apply pinhole projection with perspective division
        # - camera_intrinsics[0, 0] = fx (focal length in x direction, pixels)
        # - camera_intrinsics[0, 2] = cx (principal point x, pixels)
        # - Divide by z for perspective: closer objects (small z) project to larger pixels
        u = (
            camera_intrinsics[0, 0] * (x / z) + camera_intrinsics[0, 2]
        )  # Pixel x coordinate

        # - camera_intrinsics[1, 1] = fy (focal length in y direction, pixels)
        # - camera_intrinsics[1, 2] = cy (principal point y, pixels)
        v = (
            camera_intrinsics[1, 1] * (y / z) + camera_intrinsics[1, 2]
        )  # Pixel y coordinate

        # Store 2D pixel coordinates
        points_2d[i] = [u, v]  # Pixel coordinates (may be outside image bounds)

    # Debug output: verify 2D projection results
    if debug:
        print(f"Camera intrinsics K:\n{camera_intrinsics}")
        print(f"2D pixel coordinates sample (first 3 points): {points_2d[:3]}")
        print(f"  Format: [u, v] where u is horizontal (x), v is vertical (y)")
        print(
            f"  Image bounds check: u in [0, {DEFAULT_IMG_WIDTH}], v in [0, {DEFAULT_IMG_HEIGHT}]"
        )

    # Return 2D pixel coordinates
    # - Shape: (N, 2) where each row is [u, v] in pixels
    # - u: horizontal pixel coordinate (0 = left, increases right)
    # - v: vertical pixel coordinate (0 = top, increases down)
    # - Values may be outside [0, width] x [0, height] if point projects outside image
    return points_2d  # Shape: (N, 2)


def compute_bbox_from_2d_points(
    points_2d: np.ndarray,
    img_width: int,
    img_height: int,
    padding: float = BBOX_PADDING,
) -> Optional[BBox]:
    """
    Compute normalized bounding box from 2D joint points.

    Takes all projected 2D hand joint points and computes the smallest axis-aligned
    bounding box that contains them. Adds padding for robustness and normalizes
    coordinates to [0, 1] range for EPIC format compatibility.

    Args:
        points_2d: (N, 2) array of 2D pixel coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        padding: Expansion factor (0.1 = 10% expansion on each side)

    Returns:
        BBox object with normalized coordinates (left, top, right, bottom) in [0,1],
        or None if invalid bbox (empty, negative dimensions, or out of bounds)
    """
    # Check if we have any points to process
    if len(points_2d) == 0:
        return None  # No points, cannot compute bbox

    # =========================================================================
    # STEP 1: Find axis-aligned bounding box in pixel coordinates
    # =========================================================================
    # - Find minimum and maximum x, y coordinates across all joint points
    # - This gives the smallest rectangle that contains all points
    # - Axis-aligned: sides parallel to image edges (not rotated)

    # Find horizontal bounds (x-axis, left-right)
    x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])  # In pixels

    # Find vertical bounds (y-axis, top-bottom)
    y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])  # In pixels

    # =========================================================================
    # STEP 2: Add padding to expand bounding box
    # =========================================================================
    # - Padding prevents clipping hand edges in the bounding box
    # - Expand by padding% of current width/height on EACH side
    # - Example: if bbox is 100px wide and padding=0.1, add 10px on left AND right
    #   (total width becomes 120px)

    # Calculate current bbox dimensions
    w, h = x_max - x_min, y_max - y_min  # Width and height in pixels

    # Expand bbox by padding amount on each side
    # - Left side: move x_min left by w*padding
    # - Right side: move x_max right by w*padding
    # - Top side: move y_min up by h*padding
    # - Bottom side: move y_max down by h*padding
    # - Clamp to image boundaries: [0, img_width] x [0, img_height]
    x_min = max(0, x_min - w * padding)  # Clamp to image left edge
    x_max = min(img_width, x_max + w * padding)  # Clamp to image right edge
    y_min = max(0, y_min - h * padding)  # Clamp to image top edge
    y_max = min(img_height, y_max + h * padding)  # Clamp to image bottom edge

    # =========================================================================
    # STEP 3: Normalize coordinates to [0, 1] range (EPIC format requirement)
    # =========================================================================
    # - EPIC-KITCHENS format requires normalized bbox coordinates
    # - Normalized coords are resolution-independent (work for any image size)
    # - Formula: normalized = pixel / image_dimension
    # - Range: 0.0 (left/top edge) to 1.0 (right/bottom edge)

    left = x_min / img_width  # Normalized left edge (0 = leftmost)
    right = x_max / img_width  # Normalized right edge (1 = rightmost)
    top = y_min / img_height  # Normalized top edge (0 = topmost)
    bottom = y_max / img_height  # Normalized bottom edge (1 = bottommost)

    # =========================================================================
    # STEP 4: Validate bounding box
    # After clamping, coordinates are guaranteed to be in [0, 1], so only
    # check that the bbox has positive dimensions (left < right, top < bottom).
    if left >= right or top >= bottom:
        return None

    # Create and return EPIC BBox object
    # - BBox format: (left, top, right, bottom) all in [0, 1]
    # - Example: BBox(0.1, 0.2, 0.5, 0.7) means:
    #   - Left at 10% of width, Top at 20% of height
    #   - Right at 50% of width, Bottom at 70% of height
    return BBox(left=left, top=top, right=right, bottom=bottom)


def process_single_hand(
    hand_data: dict,
    hand_side: str,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    img_width: int,
    img_height: int,
    debug: bool = False,
) -> Optional[HandDetection]:
    """
    Process a single hand's skeletal data to generate EPIC hand detection.

    Pipeline:
    1. Extract device pose (where Vision Pro is in world)
    2. Extract 3D hand joints in world coordinates
    3. Project joints to 2D image coordinates
    4. Compute bounding box from projected joints
    5. Create EPIC HandDetection object

    Args:
        hand_data: Dictionary containing skeleton response for one hand
        hand_side: 'left' or 'right'
        camera_intrinsics: 3x3 camera intrinsic matrix
        camera_extrinsics: 4x4 camera extrinsic matrix
        img_width: Image width in pixels
        img_height: Image height in pixels
        debug: If True, print detailed processing info

    Returns:
        HandDetection object with bbox and metadata, or None if processing fails
    """
    # =========================================================================
    # STEP 1: Validate input data
    # =========================================================================
    # Check if hand tracking data is available for this frame
    # - hand_data may be None if hand is not visible
    # - response may be None if tracking failed
    if hand_data is None or hand_data.get("response") is None:
        return None  # No hand data available, skip this hand

    # Extract skeleton data from protobuf response
    skeleton = hand_data["response"]  # Protobuf skeleton response object

    try:
        # =====================================================================
        # STEP 2: Extract device pose (Vision Pro location in world)
        # =====================================================================
        # Get 4x4 transformation matrix: world_T_device
        # - Represents where the Vision Pro headset is located and oriented in world space
        # - Needed to chain transformations: world => device => camera
        device_extrinsics = get_device_pose_from_skeleton(skeleton)

        # =====================================================================
        # STEP 3: Extract 3D hand joints in world coordinates
        # =====================================================================
        # Returns (26, 3) array of hand joint positions
        # - 26 joints: 1 anchor + 25 ARKit hand joints
        # - 3 coordinates: (x, y, z) in meters, world frame
        # - World frame: ARKit's coordinate system
        joints_3d = extract_hand_joints_3d(skeleton)

        # Debug: Show joint extraction results
        if debug:
            print(f"{hand_side} hand: extracted {len(joints_3d)} joints")
            print(
                f"{hand_side} hand: 3D joint range: [{joints_3d.min():.3f}m, {joints_3d.max():.3f}m]"
            )

        # =====================================================================
        # STEP 4: Project 3D joints to 2D image coordinates
        # =====================================================================
        # Transform: World (3D) => Device (3D) => Camera (3D) => Image (2D)
        # Returns (26, 2) array of pixel coordinates [u, v]
        joints_2d = project_3d_to_2d(
            joints_3d,  # (26, 3) world coordinates
            camera_intrinsics,  # (3, 3) K matrix
            camera_extrinsics,  # (4, 4) camera pose (stored transposed)
            device_extrinsics,  # (4, 4) world_T_device
            debug=debug,
        )

        # Debug: Show projection results
        if debug:
            print(
                f"{hand_side} hand: 2D projection range: "
                f"u=[{joints_2d[:, 0].min():.1f}px, {joints_2d[:, 0].max():.1f}px], "
                f"v=[{joints_2d[:, 1].min():.1f}px, {joints_2d[:, 1].max():.1f}px]"
            )

        # =====================================================================
        # STEP 5: Compute bounding box from 2D joint projections
        # =====================================================================
        # Returns BBox with normalized coordinates [0, 1] or None if invalid
        bbox = compute_bbox_from_2d_points(joints_2d, img_width, img_height)

        # Check if bbox computation was successful
        if bbox is None:
            if debug:
                print(
                    f"{hand_side} hand: bbox computation returned None (invalid bbox)"
                )
            return None  # Invalid bbox, skip this hand

        # Debug: Show bbox coordinates
        if debug:
            print(
                f"{hand_side} hand: bbox (normalized) = "
                f"[left={bbox.left:.3f}, top={bbox.top:.3f}, "
                f"right={bbox.right:.3f}, bottom={bbox.bottom:.3f}]"
            )

        # =====================================================================
        # STEP 6: Create EPIC HandDetection object
        # =====================================================================
        # EPIC HandDetection format specification:
        # - bbox: BBox with normalized [0,1] coordinates (left, top, right, bottom)
        # - score: float32 confidence score [0,1] (1.0 = perfect, ground truth)
        # - state: HandState enum (NO_CONTACT, CONTACT, PORTABLE_OBJECT, STATIC_OBJECT)
        # - side: HandSide enum (LEFT, RIGHT)
        # - object_offset: FloatVector(x, y) offset to held object (0,0 if no object)
        return HandDetection(
            bbox=bbox,
            score=np.float32(
                1.0
            ),  # Perfect confidence (ground truth from ARKit tracking)
            state=HandState.NO_CONTACT,  # Default state (we don't have contact info from ARKit)
            side=(
                HandSide.LEFT if hand_side == "left" else HandSide.RIGHT
            ),  # LEFT or RIGHT enum
            object_offset=FloatVector(
                x=np.float32(0.0),  # No horizontal offset (not holding object)
                y=np.float32(0.0),  # No vertical offset
            ),
        )

    except Exception as e:
        # Handle any errors during processing (projection, bbox computation, etc.)
        if debug:
            print(
                f"{hand_side} hand: ERROR during processing - {type(e).__name__}: {e}"
            )
        return None  # Processing failed, skip this hand


def process_tri_episode(
    episode_path: str,
    output_dir: str,
    img_width: int = DEFAULT_IMG_WIDTH,
    img_height: int = DEFAULT_IMG_HEIGHT,
) -> Tuple[Dict[str, List[HandDetection]], np.ndarray, np.ndarray]:
    """
    Convert TRI episode.pkl (3D hand skeletal data) to EPIC hand_det.pkl (2D bboxes).

    Processes each frame in the episode:
    - Loads hand skeletal data and camera parameters
    - Projects 3D hand joints to 2D image coordinates
    - Computes bounding boxes for left and right hands
    - Saves in EPIC format: dict[frame_id -> List[HandDetection]]

    Args:
        episode_path: Path to TRI episode.pkl file (gzipped)
        output_dir: Directory to save output hand_det.pkl
        img_width: Image width (default 1920)
        img_height: Image height (default 1080)

    Returns:
        Tuple of (hand_det_data, first_frame_intrinsics, first_frame_extrinsics)
    """
    print(f"Loading: {episode_path}")

    # =========================================================================
    # STEP 1: Load episode data
    # =========================================================================
    # TRI episode.pkl format:
    # - File: gzip-compressed pickle file
    # - Structure: dictionary with keys:
    #   - 'frame_timestamps': list of timestamps (seconds)
    #   - 'pose_snapshots': list of dicts, each containing 'left' and 'right' hand data
    #   - 'camera_intrinsics': list of (3,3) K matrices per frame
    #   - 'camera_extrinsics': list of (4,4) camera pose matrices per frame
    #   - 'success': boolean indicating if episode was successful
    with gzip.open(episode_path, "rb") as f:
        episode_data = pickle.load(f)  # Load pickled dictionary

    # =========================================================================
    # STEP 2: Print episode metadata summary
    # =========================================================================
    total_frames = len(
        episode_data["frame_timestamps"]
    )  # Number of frames in episode
    start_time = episode_data["frame_timestamps"][0]  # First frame timestamp
    end_time = episode_data["frame_timestamps"][-1]  # Last frame timestamp
    duration = end_time - start_time  # Episode duration (seconds)

    print(f"Episode info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Success: {episode_data['success']}")
    print(f"  - Duration: {duration:.2f}s")

    # =========================================================================
    # STEP 3: Initialize output data structures
    # =========================================================================
    # EPIC hand_det.pkl format:
    # - Type: dict[str, List[HandDetection]]
    # - Key: frame index as string ("0", "1", "2", ...)
    # - Value: list of HandDetection objects (0-2 hands per frame)
    hand_det_data = (
        {}
    )  # Output dictionary: frame_id (str) -> List[HandDetection]

    # Track failures for statistics
    failed_frames = {
        "left": 0,
        "right": 0,
    }  # Count failed hand processings per side

    # Debug: print detailed info for first frame only
    debug_frame = 0  # Frame index to debug (0 = first frame)

    # Store first frame camera params for JSON export
    # - Used to save camera calibration in separate JSON files
    # - First frame params are representative (params don't change much across frames)
    first_frame_intrinsics = episode_data["camera_intrinsics"][
        0
    ]  # (3, 3) K matrix
    first_frame_extrinsics = episode_data["camera_extrinsics"][
        0
    ]  # (4, 4) pose matrix

    # =========================================================================
    # STEP 4: Process each frame in the episode
    # =========================================================================
    # Iterate through all pose snapshots (one per frame)
    # - Each snapshot contains left/right hand skeletal data
    # - Camera parameters may vary per frame (device movement)
    for frame_idx, pose_snapshot in enumerate(episode_data["pose_snapshots"]):
        # =====================================================================
        # Get camera parameters for current frame
        # =====================================================================
        # - Intrinsics: (3, 3) camera matrix K [fx, fy, cx, cy]
        # - Extrinsics: (4, 4) camera pose matrix (stored transposed)
        # Note: Camera params can change frame-to-frame as Vision Pro moves
        camera_intrinsics = episode_data["camera_intrinsics"][
            frame_idx
        ]  # (3, 3)
        camera_extrinsics = episode_data["camera_extrinsics"][
            frame_idx
        ]  # (4, 4)

        # Print debug header for first frame
        if frame_idx == debug_frame:
            print(f"\nDebug frame {debug_frame}:")

        # Initialize list to store hand detections for this frame
        # - Can contain 0, 1, or 2 HandDetection objects (left/right hands)
        frame_detections = []  # List[HandDetection]

        # =====================================================================
        # Process both hands (left and right)
        # =====================================================================
        for hand_side in ["left", "right"]:
            # Get hand data from pose snapshot
            # - May be None if hand is not visible in this frame
            # - Dictionary with 'response' key containing protobuf skeleton data
            hand_data = pose_snapshot.get(hand_side)  # dict or None

            # Process hand: 3D joints => 2D joints => bbox => HandDetection
            # Returns HandDetection object or None if processing fails
            detection = process_single_hand(
                hand_data,  # Hand skeletal data (or None)
                hand_side,  # "left" or "right"
                camera_intrinsics,  # (3, 3) K matrix
                camera_extrinsics,  # (4, 4) camera pose
                img_width,  # Image width for bbox computation
                img_height,  # Image height for bbox computation
                debug=(
                    frame_idx == debug_frame
                ),  # Enable debug for first frame
            )

            # Collect successful detections
            if detection is not None:
                frame_detections.append(
                    detection
                )  # Add to frame's detection list
            elif (
                hand_data is not None and hand_data.get("response") is not None
            ):
                # Hand data exists but processing failed
                # - Possible causes: invalid projection, bbox out of bounds, etc.
                failed_frames[hand_side] += 1  # Increment failure counter

        # =====================================================================
        # Store frame detections using string frame index as key
        # =====================================================================
        # EPIC format requirement: frame indices must be strings
        # - "0", "1", "2", ... (not integers)
        # - Empty list if no hands detected in frame
        hand_det_data[str(frame_idx)] = (
            frame_detections  # Map "frame_id" => [detections]
        )
    # =========================================================================
    # STEP 5: Calculate and print statistics
    # =========================================================================
    # Count frames where each hand was successfully detected
    frames_with_left = sum(
        1  # Count 1 for each frame
        for dets in hand_det_data.values()  # Iterate through all frame detections
        if any(
            d.side == HandSide.LEFT for d in dets
        )  # If any detection is LEFT hand
    )
    frames_with_right = sum(
        1  # Count 1 for each frame
        for dets in hand_det_data.values()  # Iterate through all frame detections
        if any(
            d.side == HandSide.RIGHT for d in dets
        )  # If any detection is RIGHT hand
    )

    # Print conversion results summary
    print(f"\nResults:")
    print(f"  Total frames in hand_det.pkl: {len(hand_det_data)}")
    print(f"  Left hand detected: {frames_with_left} frames")
    print(f"  Right hand detected: {frames_with_right} frames")

    # Count empty frames (no hands detected)
    empty_frames = sum(1 for dets in hand_det_data.values() if len(dets) == 0)
    print(f"  Empty frames (no hands): {empty_frames} frames")

    # Print failure statistics if any failures occurred
    if failed_frames["left"] or failed_frames["right"]:
        print(
            f"  Failed processing: left={failed_frames['left']}, right={failed_frames['right']}"
        )

    # =========================================================================
    # STEP 6: Validate frame key consistency
    # =========================================================================
    # Ensure frame indices are sequential: "0", "1", "2", ..., "N-1"
    # - EPIC format expects continuous frame indices starting from 0
    expected_keys = set(
        str(i) for i in range(len(hand_det_data))
    )  # {"0", "1", ...}
    actual_keys = set(hand_det_data.keys())  # Keys actually present in output

    if expected_keys != actual_keys:
        print(f"  WARNING: Frame keys are not sequential!")

        # Print sample of missing keys if any
        missing = (
            expected_keys - actual_keys
        )  # Keys that should exist but don't
        if missing:
            # Sort numerically and show first 10 missing keys
            missing_sorted = sorted(missing, key=int)[:10]
            print(f"Missing keys: {missing_sorted}...")

    # =========================================================================
    # STEP 7: Save output file
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if needed
    output_path = os.path.join(output_dir, "hand_det.pkl")  # hand_det.pkl path

    # Save as standard pickle (not gzipped)
    # - EPIC format uses uncompressed pickle for hand_det.pkl
    # - Binary mode: 'wb'
    with open(output_path, "wb") as f:
        pickle.dump(hand_det_data, f)  # Serialize dictionary to file

    print(f"\nSaved: {output_path}")

    return hand_det_data, first_frame_intrinsics, first_frame_extrinsics


def validate_output(output_path: str) -> bool:
    """
    Validate the generated hand_det.pkl file structure and content.

    Checks:
    - File exists and loads successfully
    - Data is a non-empty dictionary
    - Detections have required attributes (side, bbox)
    - Bounding boxes have valid normalized coordinates

    Args:
        output_path: Path to hand_det.pkl file

    Returns:
        True if validation passes, False otherwise
    """
    if not os.path.exists(output_path):
        print(f"Error: File not found")
        return False

    try:
        # Load and check basic structure
        with open(output_path, "rb") as f:
            data = pickle.load(f)

        # Check data structure
        if not isinstance(data, dict) or len(data) == 0:
            print(f"Error: Invalid data structure")
            return False

        # Sample first frame for validation
        first_key = list(data.keys())[0]
        first_dets = data[first_key]

        print(f"\nValidation:")
        print(f"  Total frames: {len(data)}")
        print(f"  First frame: '{first_key}' ({len(first_dets)} detections)")

        # If there are detections, validate their structure
        if len(first_dets) > 0:
            det = first_dets[0]
            # Check required attributes
            if not hasattr(det, "side") or not hasattr(det, "bbox"):
                print(f"Error: Missing required attributes")
                return False

            # Check bbox values are normalized and valid
            bbox = det.bbox
            if not (
                0 <= bbox.left < bbox.right <= 1
                and 0 <= bbox.top < bbox.bottom <= 1
            ):
                print(f"Error: Invalid bbox values")
                return False

            print(
                f"  Example: {det.side.name} [{bbox.left:.3f}, {bbox.top:.3f}, {bbox.right:.3f}, {bbox.bottom:.3f}]"
            )

        print(f"\nValidation passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def downsample_video(
    video_src: str,
    output_dir: str,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    output_filename: str = "video_L.mp4",
) -> None:
    """
    Downsample video to target resolution.

    The Phantom pipeline's E2FGVI inpaint model cannot handle 1920x1080 in EPIC mode
    (epic=true means no resize in _load_and_prepare_frames => GPU OOM).
    EPIC-KITCHENS videos are 456x256, so we downsample to match.

    Args:
        video_src: Path to source video (e.g., main_camera.mp4)
        output_dir: Output directory for downsampled video
        target_w: Target width in pixels
        target_h: Target height in pixels
        output_filename: Output filename
    """
    import cv2  # OpenCV for video processing

    # =========================================================================
    # STEP 1: Validate input and open video
    # =========================================================================
    if not os.path.exists(video_src):
        print(f"  Warning: Video not found: {video_src}")
        return  # Cannot proceed without source video

    # Open video capture object
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"  Error: Cannot open video: {video_src}")
        return  # Failed to open video file

    # =========================================================================
    # STEP 2: Read video metadata
    # =========================================================================
    # Extract properties from source video using OpenCV property IDs
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Original width (pixels)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Original height (pixels)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frame count

    # Print video information
    print(f"  Source: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames")
    print(f"  Target: {target_w}x{target_h}")
    print(
        f"  Scale:  {target_w/orig_w:.4f}x (width), {target_h/orig_h:.4f}x (height)"
    )

    # =========================================================================
    # STEP 3: Create output video writer
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    output_path = os.path.join(output_dir, output_filename)

    # Setup video codec and writer
    # - fourcc: "Four Character Code" for video compression format
    # - 'mp4v': MPEG-4 video codec (widely compatible)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 video codec

    # Create VideoWriter object
    # - Parameters: (filename, codec, fps, frame_size)
    # - frame_size must be (width, height) tuple
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    # =========================================================================
    # STEP 4: Process video frame-by-frame
    # =========================================================================
    frame_count = 0  # Track number of processed frames

    while True:
        # Read next frame from video
        # - ret: boolean indicating success
        # - frame: numpy array (H, W, 3) in BGR format
        ret, frame = cap.read()

        if not ret:
            break  # End of video or read error

        # Resize frame to target resolution
        # - cv2.resize(src, dsize, interpolation)
        # - INTER_AREA: best for downsampling (averages pixels)
        # - Alternative: INTER_LINEAR (faster but lower quality)
        resized = cv2.resize(
            frame,  # Source frame (orig_w, orig_h, 3)
            (target_w, target_h),  # Target size (width, height)
            interpolation=cv2.INTER_AREA,  # Area interpolation for downsampling
        )

        # Write resized frame to output video
        writer.write(resized)  # Append frame to video file
        frame_count += 1  # Increment counter

    # =========================================================================
    # STEP 5: Cleanup and report results
    # =========================================================================
    cap.release()  # Close input video
    writer.release()  # Finalize and close output video

    # Calculate output file size in megabytes
    file_size = os.path.getsize(output_path) / (
        1024 * 1024
    )  # Convert bytes to MB

    print(f"  => {output_path} ({frame_count} frames, {file_size:.1f} MB)")


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
    Scale camera intrinsics to match downsampled video resolution and save in Phantom format.

    Args:
        intrinsics: (3, 3) original camera intrinsic matrix
        orig_w, orig_h: Original video dimensions
        target_w, target_h: Target (downsampled) video dimensions
        output_dir: Output directory
        output_filename: Output filename
    """
    # =========================================================================
    # STEP 1: Calculate scaling factors
    # =========================================================================
    # When video is downsampled, focal lengths and principal point must scale proportionally
    # Example: 1920x1080 => 456x256
    # - scale_x = 456/1920 = 0.2375 (horizontal scaling)
    # - scale_y = 256/1080 = 0.2370 (vertical scaling)
    scale_x = target_w / orig_w  # Horizontal scale factor
    scale_y = target_h / orig_h  # Vertical scale factor

    # =========================================================================
    # STEP 2: Scale intrinsic parameters
    # =========================================================================
    # Original intrinsic matrix K:
    #   [[fx,  0, cx],
    #    [ 0, fy, cy],
    #    [ 0,  0,  1]]
    #
    # Scaling rules:
    # - fx_new = fx_old * scale_x (focal length scales with width)
    # - fy_new = fy_old * scale_y (focal length scales with height)
    # - cx_new = cx_old * scale_x (principal point x scales with width)
    # - cy_new = cy_old * scale_y (principal point y scales with height)
    #
    # Why: If image shrinks 2x, all pixel measurements shrink 2x

    fx = float(intrinsics[0, 0]) * scale_x  # Scaled focal length X (pixels)
    fy = float(intrinsics[1, 1]) * scale_y  # Scaled focal length Y (pixels)
    cx = float(intrinsics[0, 2]) * scale_x  # Scaled principal point X (pixels)
    cy = float(intrinsics[1, 2]) * scale_y  # Scaled principal point Y (pixels)

    # =========================================================================
    # STEP 3: Calculate field of view (FOV) for target resolution
    # =========================================================================
    # FOV formula: FOV = 2 * arctan(image_dimension / (2 * focal_length))
    # - Larger FOV = wider viewing angle
    # - FOV depends on focal length and sensor size (image dimensions)
    # - Convert from radians to degrees (* 180/π)

    # Horizontal FOV (field of view in X direction)
    h_fov = 2 * np.arctan(target_w / (2 * fx)) * 180 / np.pi  # Degrees

    # Vertical FOV (field of view in Y direction)
    v_fov = 2 * np.arctan(target_h / (2 * fy)) * 180 / np.pi  # Degrees

    # Diagonal FOV (field of view along image diagonal)
    diagonal = np.sqrt(target_w**2 + target_h**2)  # Diagonal length in pixels
    d_fov = 2 * np.arctan(diagonal / (2 * fx)) * 180 / np.pi  # Degrees

    # =========================================================================
    # STEP 4: Build Phantom camera format JSON structure
    # =========================================================================
    # Phantom expects this specific format:
    # {
    #   "left": {fx, fy, cx, cy, disto, v_fov, h_fov, d_fov},
    #   "right": {same structure}
    # }
    # Note: "left" and "right" refer to stereo camera systems, but we use same params for both

    cam_entry = {
        "fx": fx,  # Focal length X in pixels (scaled)
        "fy": fy,  # Focal length Y in pixels (scaled)
        "cx": cx,  # Principal point X in pixels (scaled)
        "cy": cy,  # Principal point Y in pixels (scaled)
        "disto": [0.0]
        * 12,  # Distortion coefficients (12 zeros = no distortion)
        "v_fov": float(v_fov),  # Vertical field of view (degrees)
        "h_fov": float(h_fov),  # Horizontal field of view (degrees)
        "d_fov": float(d_fov),  # Diagonal field of view (degrees)
    }

    # Create symmetric structure (same params for left/right)
    camera_params = {
        "left": cam_entry,
        "right": dict(cam_entry),  # Copy for right camera
    }

    # =========================================================================
    # STEP 5: Save to JSON file
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)  # Create directory if needed
    output_path = os.path.join(output_dir, output_filename)

    # Write JSON with indentation for readability
    with open(output_path, "w") as f:
        json.dump(camera_params, f, indent=2)

    # Print summary of saved parameters
    print(
        f"  Scaled intrinsics: fx={fx:.2f}px, fy={fy:.2f}px, cx={cx:.2f}px, cy={cy:.2f}px"
    )
    print(f"  => {output_path}")


def save_extrinsics_to_json(
    extrinsics: np.ndarray = None,  # Not used, kept for API compatibility
    output_dir: str = "/data/maxshen/phantom/phantom/camera",
    output_filename: str = "camera_extrinsics_tri.json",
) -> None:
    """
    Save camera extrinsics to JSON in Phantom format.

    For TRI egocentric data, we reuse Phantom's EPIC ego bimanual shoulders
    extrinsics. This is because:
    - Phantom's action_processor uses T_cam2robot to map HaMeR's camera-frame
      3D hand keypoints into a "robot frame" that BASE_T_1 then maps to MuJoCo.
    - The EPIC extrinsics were calibrated so that the resulting robot-frame
      positions fall within Kinova3's reachable workspace.
    - TRI egocentric data has the same viewing geometry (head-mounted,
      looking down at hands on table), so the same T_cam2robot applies.
    - Using identity matrix would leave points in camera frame (Z≈0.5m forward),
      which after BASE_T_1 lands outside the robot workspace => tracking errors.

    Args:
        extrinsics: IGNORED - kept for backward compatibility
        output_dir: Output directory
        output_filename: Output filename
    """
    import shutil

    # Copy the proven EPIC ego bimanual shoulders extrinsics
    src = os.path.join(
        output_dir, "camera_extrinsics_ego_bimanual_shoulders.json"
    )
    dst = os.path.join(output_dir, output_filename)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Camera extrinsics copied from EPIC ego bimanual shoulders")
        print(f"  => {dst}")
    else:
        # Fallback: hardcode the EPIC ego bimanual shoulders values
        extrinsics_data = [
            {
                "num_marker_seen": 114,
                "stage2_retry": 11,
                "pixel_error": 2.1157278874907863,
                "proj_func": "hand_marker_proj_world_camera",
                "camera_base_ori": [
                    [
                        -0.7220417114840215,
                        0.37764981440725887,
                        0.579686453658689,
                    ],
                    [
                        0.020370475586732495,
                        0.8491206965938227,
                        -0.527805917303316,
                    ],
                    [
                        -0.6915495720493177,
                        -0.3692893991088662,
                        -0.6207934673498243,
                    ],
                ],
                "camera_base_ori_rotvec": [
                    0.2877344548443808,
                    2.3075097094104504,
                    -0.6485227972051454,
                ],
                "camera_base_pos": [
                    -0.5123627783256401,
                    -0.11387480700266536,
                    0.3151264229148423,
                ],
                "p_marker_ee": [
                    -0.041990731174163416,
                    -0.02636865486252487,
                    -0.01442948433864288,
                ],
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
        print(
            f"  Camera extrinsics saved (EPIC ego bimanual shoulders, hardcoded)"
        )
        print(f"  => {dst}")


def _sort_key_for_dir(d: str):
    try:
        return int(d.split("_", 1)[1].split()[0])
    except (IndexError, ValueError):
        return d


def find_all_episodes(base_dir: str) -> List[str]:
    """
    Recursively find all episode.pkl files under base_dir.

    Args:
        base_dir: Root directory to search (e.g. /data/maxshen/Video_data/LBM_human_egocentric)

    Returns:
        Sorted list of absolute paths to episode.pkl files
    """
    episode_paths = []
    for root, dirs, files in os.walk(base_dir):
        # Sort dirs in-place so os.walk traverses them in order
        dirs.sort(key=_sort_key_for_dir)
        for fname in sorted(files):
            if fname == "episode.pkl":
                episode_paths.append(os.path.join(root, fname))
    return episode_paths


def process_single_episode(
    episode_path: str,
    output_dir: str,
    save_camera_params: bool = False,
) -> bool:
    """
    Process one TRI episode end-to-end:
    1. Convert hand detections to EPIC format (hand_det.pkl)
    2. Downsample video to 456x256 (video_L.mp4)
    3. Optionally save camera intrinsics / extrinsics JSON files

    Args:
        episode_path: Absolute path to episode.pkl (gzipped)
        output_dir:   Directory where outputs are written
        save_camera_params: If True, also write camera_intrinsics_tri.json
                            and camera_extrinsics_tri.json to the Phantom
                            camera directory (only needed once per run)

    Returns:
        True if hand_det.pkl passes validation, False otherwise
    """
    video_path = os.path.join(os.path.dirname(episode_path), "main_camera.mp4")

    # ------------------------------------------------------------------
    # STEP 1: Generate hand_det.pkl
    # ------------------------------------------------------------------
    try:
        _, first_intrinsics, first_extrinsics = process_tri_episode(
            episode_path, output_dir
        )
    except Exception as e:
        print(f"\n  ERROR during Generate hand_det.pkl: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # STEP 2: Downsample video
    # ------------------------------------------------------------------
    if os.path.exists(video_path):
        try:
            downsample_video(
                video_path, output_dir, TARGET_WIDTH, TARGET_HEIGHT
            )
        except Exception as e:
            print(f"\n  ERROR during Downsample video: {e}")
            import traceback

            traceback.print_exc()
            return False
    else:
        print(f"  Warning: Video not found at {video_path}")

    # ------------------------------------------------------------------
    # STEP 3 (optional): Save camera intrinsics / extrinsics JSONs
    # ------------------------------------------------------------------
    if save_camera_params:
        save_scaled_intrinsics(
            first_intrinsics,
            DEFAULT_IMG_WIDTH,
            DEFAULT_IMG_HEIGHT,
            TARGET_WIDTH,
            TARGET_HEIGHT,
        )
        save_extrinsics_to_json(first_extrinsics)

    # ------------------------------------------------------------------
    # Validate output
    # ------------------------------------------------------------------
    return validate_output(os.path.join(output_dir, "hand_det.pkl"))


def main():
    """
    Main entry point for TRI to EPIC batch conversion.

    Scans all episode.pkl files under INPUT_BASE_DIR and converts each one
    to EPIC format, writing output to numbered sub-directories under
    OUTPUT_BASE_DIR (0, 1, 2, …).

    Already-processed episodes (output directory already contains
    hand_det.pkl) are skipped unless FORCE_REPROCESS is set to True.

    Generates per episode:
    1. hand_det.pkl  - 2D bounding boxes from 3D hand skeletons
    2. video_L.mp4   - downsampled to 456x256 for E2FGVI compatibility

    Generates once (from the first episode):
    3. camera_intrinsics_tri.json - scaled intrinsics in Phantom format
    4. camera_extrinsics_tri.json - reuses EPIC ego bimanual shoulders extrinsics
    """
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    # Root directory that contains all TRI episodes
    INPUT_BASE_DIR = "/data/maxshen/Video_data/LBM_human_egocentric/egoPutKiwiInCenterOfTable/2025-11-13_12-46-27"

    # Root output directory; each episode gets a numbered sub-directory
    OUTPUT_BASE_DIR = "/data/maxshen/phantom/data/raw/tri"

    # Set to True to re-convert episodes that already have hand_det.pkl
    FORCE_REPROCESS = False

    # =========================================================================
    # DISCOVER ALL EPISODES
    # =========================================================================
    print("=" * 60)
    print("TRI to EPIC Hand Detection Converter  (batch mode)")
    print("=" * 60)
    print(f"\nSearching for episodes in: {INPUT_BASE_DIR}")

    episode_paths = find_all_episodes(INPUT_BASE_DIR)

    if not episode_paths:
        print("Error: No episode.pkl files found.")
        return 1

    print(f"Found {len(episode_paths)} episode(s).\n")

    # =========================================================================
    # BATCH PROCESSING LOOP
    # =========================================================================
    results = {"success": 0, "skipped": 0, "failed": 0}
    save_camera_params_done = False  # Only save camera JSONs once

    for idx, episode_path in enumerate(episode_paths[:1]):
        output_dir = os.path.join(OUTPUT_BASE_DIR, str(idx))

        print("=" * 60)
        print(f"[{idx + 1}/{len(episode_paths)}] Episode index: {idx}")
        print(f"  Input : {episode_path}")
        print(f"  Output: {output_dir}")

        # ------------------------------------------------------------------
        # Skip check: if hand_det.pkl already exists, skip unless forced
        # ------------------------------------------------------------------
        hand_det_path = os.path.join(output_dir, "hand_det.pkl")
        if os.path.exists(hand_det_path) and not FORCE_REPROCESS:
            print("  => Already processed (hand_det.pkl exists), skipping.")
            results["skipped"] += 1
            continue

        # ------------------------------------------------------------------
        # Fail check: Validate input exists
        # ------------------------------------------------------------------
        if not os.path.exists(episode_path):
            print(f"  ERROR: episode.pkl not found, skipping.")
            results["failed"] += 1
            continue

        # ------------------------------------------------------------------
        # Process this episode
        # ------------------------------------------------------------------
        # Save camera params only for the very first episode that is actually
        # processed (not skipped), so the shared Phantom camera directory gets
        # populated exactly once per run.
        save_cam = not save_camera_params_done

        success = process_single_episode(episode_path, output_dir, save_cam)

        if save_cam and success:
            save_camera_params_done = True

        if success:
            results["success"] += 1
        else:
            results["failed"] += 1

    # =========================================================================
    # SUMMARY
    # =========================================================================
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
    print("  python process_data.py demo_name=tri mode=all --config-name=tri")
    print("=" * 60)

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
