import os
import sys
import numpy as np
import json
import yaml
import pickle
import gzip
from collections import defaultdict
from glob import glob
from tqdm import tqdm, trange
import pandas as pd
import cv2
import imageio
from scipy.spatial.transform import Rotation
from copy import copy, deepcopy


def draw_points_and_line(
    image,
    pt1,
    pt2,
    circle_radius=5,
    circle_color=(0, 0, 255),
    line_color=(255, 0, 0),
    thickness=2,
):
    """
    Draws a circle at each of two points and a line connecting them on an image.

    Args:
        image (numpy.ndarray): The input image (BGR).
        pt1 (tuple): First point (x, y).
        pt2 (tuple): Second point (x, y).
        circle_radius (int): Radius of the circles to draw.
        circle_color (tuple): BGR color for the circles.
        line_color (tuple): BGR color for the connecting line.
        thickness (int): Line and circle thickness.

    Returns:
        numpy.ndarray: The image with the drawings.
    """
    # Draw circles
    cv2.circle(image, pt1, circle_radius, circle_color, -1)  # filled circle
    cv2.circle(image, pt2, circle_radius, circle_color, -1)

    # Draw line connecting them
    cv2.line(image, pt1, pt2, line_color, thickness)

    return image


def project_point_to_image_corrected(
    point_T, device_extrinsics, camera_extrinsics, K
):
    """
    Project a 3D point in world coordinates to camera image coordinates.

    Args:
        point_world: 3D position in world frame (3,) array
        device_extrinsics: 4x4 world_T_device (device pose in world)
        camera_extrinsics: 4x4 device_T_camera (camera pose relative to device)
        K: 3x3 camera intrinsics

    Returns:
        [u, v] pixel coordinates or None if behind camera
    """
    # The camera_extrinsics has translation in bottom row, so transpose it
    device_T_camera = camera_extrinsics.T
    device_T_camera = np.linalg.inv(device_T_camera)

    point_world = point_T[:3, 3]

    # Compute world_T_camera
    world_T_camera = device_extrinsics @ device_T_camera

    # Transform point to camera frame
    camera_T_world = np.linalg.inv(world_T_camera)
    p_cam_h = camera_T_world @ np.hstack([point_world, 1])
    x_cam, y_cam, z_cam = p_cam_h[:3]

    # # Check if point is in front of camera (negative Z in Apple Vision Pro convention)
    # if z_cam >= 0:
    #     return None  # Behind camera

    # Project to image plane
    u = K[0, 0] * (x_cam / z_cam) + K[0, 2]
    v = K[1, 1] * (y_cam / z_cam) + K[1, 2]

    return np.array([u, v])


def transform_point_to_camera_frame_xyz(
    point_T, device_extrinsics, camera_extrinsics
):
    """
    Project a 3D point in world coordinates to camera image coordinates.

    Args:
        point_world: 3D position in world frame (3,) array
        device_extrinsics: 4x4 world_T_device (device pose in world)
        camera_extrinsics: 4x4 device_T_camera (camera pose relative to device)
        K: 3x3 camera intrinsics

    Returns:
        [u, v] pixel coordinates or None if behind camera
    """
    # The camera_extrinsics has translation in bottom row, so transpose it
    device_T_camera = camera_extrinsics.T
    device_T_camera = np.linalg.inv(device_T_camera)

    point_world = point_T[:3, 3]

    # Compute world_T_camera
    world_T_camera = device_extrinsics @ device_T_camera

    # Transform point to camera frame
    camera_T_world = np.linalg.inv(world_T_camera)
    p_cam_h = camera_T_world @ np.hstack([point_world, 1])
    return p_cam_h[:3]


def to_camera_plane(p_cam_h, K):
    x_cam, y_cam, z_cam = p_cam_h
    u = K[0, 0] * (x_cam / z_cam) + K[0, 2]
    v = K[1, 1] * (y_cam / z_cam) + K[1, 2]

    return np.array([u, v])


def parse_to_se3(pose):
    """
    Parse translation and rotation data into SE(3) format (4x4 homogeneous matrix).

    Args:
        pose: Object with pose.translation and pose.rotation attributes,
              where each has x, y, z (and w for rotation) attributes

    Returns:
        se3_matrix: 4x4 numpy array representing the SE(3) transformation
    """
    # Extract translation values
    t = np.array([pose.translation.x, pose.translation.y, pose.translation.z])

    # Extract rotation values (quaternion as x, y, z, w)
    q = np.array(
        [pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w]
    )

    # Convert quaternion (x, y, z, w) to rotation matrix
    # scipy uses (x, y, z, w) format
    rot = Rotation.from_quat(q)
    R = rot.as_matrix()

    # Create SE(3) matrix (4x4)
    se3 = np.eye(4)
    se3[:3, :3] = R
    se3[:3, 3] = t

    return se3


def load_frames_to_list(video_path):
    """
    Reads all frames from an mp4 file and returns them in a list of numpy arrays.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: video_path")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def load_dict(filepath):
    """
    Load a dictionary saved with gzip-compressed pickle.
    """
    with gzip.open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def parse_arguments():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Display left/right Vision Pro camera feeds via gRPC"
    )
    parser.add_argument("--episode-dir", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    episode_dir = args.episode_dir
    video_file = os.path.join(episode_dir, "main_camera.mp4")
    episode_file = os.path.join(episode_dir, "episode.pkl")

    print(f"Loading video frames...")
    video_frames = load_frames_to_list(video_file)

    print(f"Loading annotations...")
    episode = load_dict(episode_file)
    print(f"Done loading.")
    pose_snapshots = episode["pose_snapshots"]
    assert len(video_frames) == len(
        pose_snapshots
    ), f"len(video_frames): {len(video_frames)}, len(pose_snapshots): {len(pose_snapshots)}"

    initial_frame = deepcopy(video_frames[0])
    initial_frame = np.flip(initial_frame, axis=-1)
    initial_frame = np.array(initial_frame) # For some reason you have to do this after np.flip to avoid a cv2 error
    intrinsics_t0 = episode["camera_intrinsics"][0]
    extrinsics_t0 = episode["camera_extrinsics"][0]
    pose_snapshot_t0 = pose_snapshots[0]

    new_frames = []
    for i, frame in enumerate(video_frames):
        # frame = cv2.resize(frame, (1920, 1080))
        frame = np.flip(frame, axis=-1) # This is here because CV2 loads images in BGR format instead of RGB format
        frame = np.array(frame) # For some reason you have to do this after np.flip to avoid a cv2 error

        # imageio.imwrite("frame.png", frame))
        pose_snapshot = pose_snapshots[i]

        intrinsics = episode["camera_intrinsics"][i]
        extrinsics = episode["camera_extrinsics"][i]

        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        WHITE = (255, 255, 255)

        for handside in ["left", "right"]:
            if pose_snapshot[handside] is None:
                continue

            response = pose_snapshot[handside]["response"]
            anchor_transform = response.hand.anchor_transform
            print(f"handside: {handside}, anchor_transform: {anchor_transform}")
            anchor_transform_se3 = parse_to_se3(anchor_transform)
            print(f"anchor_transform_se3: {anchor_transform_se3}")
    #         index_finger_knuckle = response.hand.hand_skeleton.index_finger_knuckle
    #         index_finger_knuckle_se3 = parse_to_se3(index_finger_knuckle)

    #         knuckle_se3 =  anchor_transform_se3 @ index_finger_knuckle_se3

    #         device_extrinsics = parse_to_se3(response.device)
    #         anchor_pixel_coords = project_point_to_image_corrected(anchor_transform_se3, device_extrinsics, extrinsics, intrinsics).astype(int)
    #         knuckle_pixel_coords = project_point_to_image_corrected(knuckle_se3, device_extrinsics, extrinsics, intrinsics).astype(int)

    #         color = RED if handside == "left" else BLUE
    #         frame = draw_points_and_line(frame, anchor_pixel_coords, knuckle_pixel_coords, circle_radius=5, circle_color=color, line_color=color, thickness=2)
    #         frame = cv2.putText(frame, handside, knuckle_pixel_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    #         # Resize to whatever size you need for your data processing (note: resizing to small sizes can make the annotations disappear)
    #         # frame = cv2.resize(frame, (960, 540))

    #         new_frames.append(frame)

    #         # @Zeqing this part shows how to project the gripper positions into the xyz coordinate frame of the camera at t=0
    #         response_t0 = pose_snapshot_t0[handside]["response"]
    #         device_extrinsics_t0 = parse_to_se3(response_t0.device)
    #         knuckle_xyz_t0 = transform_point_to_camera_frame_xyz(knuckle_se3, device_extrinsics_t0, extrinsics_t0)

    #         # Project to the 2D camera plane at t=0 for visualization
    #         knuckle_pixel_coords_t0 = to_camera_plane(knuckle_xyz_t0, intrinsics_t0).astype(int)
    #         initial_frame = cv2.circle(initial_frame, knuckle_pixel_coords_t0, 5, color, -1)

    #         # Also visualize the xyz positions not transformed into the first camera's coordinate frame (but rather in the coordinate
    #         # frame of the camera at time t) to compare the difference.
    #         knuckle_xyz_t = transform_point_to_camera_frame_xyz(knuckle_se3, device_extrinsics, extrinsics)
    #         knuckle_pixel_coords_t = to_camera_plane(knuckle_xyz_t, intrinsics).astype(int)
    #         assert np.array_equal(knuckle_pixel_coords, knuckle_pixel_coords_t) # This two should always be equal, since it's equivalent logic to get here
    #         initial_frame = cv2.circle(initial_frame, knuckle_pixel_coords_t, 5, WHITE, -1)

    #     # cv2.imshow("Left Main Camera", frame)
    #     # cv2.waitKey(1)

    # FPS = 20
    # video_file = os.path.join(episode_dir, "main_camera_annotated.mp4")
    # imageio.mimsave(video_file, new_frames, fps=FPS, macro_block_size=1)
    # print(f"Saved episode images to {video_file}.")

    # initial_frame_file = os.path.join(episode_dir, "initial_frame_annotated.png")
    # imageio.imsave(initial_frame_file, initial_frame)
    # print(f"Saved initial frame to {initial_frame_file}.")


"""
Example usage:

python3 -u load_human_egocentric.py \
--episode-dir "/data/maxshen/LBM_human_egocentric/egoPutKiwiInCenterOfTable/2025-11-13_12-46-27/episode_46 (success)"

"""
