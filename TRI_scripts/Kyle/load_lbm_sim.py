import os 
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm, trange
import pandas as pd
import cv2 
import imageio
from copy import deepcopy


def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)  # Use safe_load to prevent code execution
    return data

def project_3d_to_pixels(point_3d, intrinsic_matrix, extrinsic_matrix):
    # intrinsic_matrix[:, :2] = intrinsic_matrix[:, :2] * 0.534
    point_3d_homogeneous = np.append(point_3d, 1)
    # point_3d_camera = np.dot(extrinsic_matrix, point_3d_homogeneous)
    point_3d_camera = np.dot(np.linalg.inv(extrinsic_matrix), point_3d_homogeneous)
    point_2d_homogeneous = np.dot(intrinsic_matrix, point_3d_camera[:3])
    point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
    return point_2d.round().astype(int)

def project_3d_to_3d(point_3d, extrinsic_matrix):
    point_3d_homogeneous = np.append(point_3d, 1)
    
    # Transform to camera frame
    point_3d_camera_homogeneous = np.dot(np.linalg.inv(extrinsic_matrix), point_3d_homogeneous)
    
    # Return the 3D coordinates (drop the homogeneous coordinate)
    return point_3d_camera_homogeneous[:3]


def project_from_3d_to_pixels(point_3d_camera, intrinsic_matrix):
    point_2d_homogeneous = np.dot(intrinsic_matrix, point_3d_camera)
    point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
    return point_2d.round().astype(int)

def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Display left/right Vision Pro camera feeds via gRPC")
    parser.add_argument("--episode-dir", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    episode_dir = args.episode_dir

    task = episode_dir.strip("/ ").split("/")[3]

    language_instructions = load_yaml("language_annotations.yaml")["language_dict"]
    possible_task_descriptions = language_instructions[task]["original"] + language_instructions[task]["randomized"]

    new_frames = []

    meta_data_file = os.path.join(episode_dir, "processed", "metadata.yaml")
    if not os.path.isfile(meta_data_file):
        print(f"{meta_data_file} does not exist. Skipping. episode_dir: {episode_dir}.") 
    else:
        meta_data = load_yaml(meta_data_file)
        camera_names = {val:key for key, val in meta_data["camera_id_to_semantic_name"].items()}

        print(f"Loading observations...")
        observations_file = os.path.join(episode_dir, "processed", "observations.npz")
        observations = np.load(observations_file)

        print(f"Loading low dim states...")
        # NOTE: hand sides are reversed intentionally here, because the original data is from a 3rd person viewpoint, so 
        # the handsides get flipped from an egocentric viewpoint 
        pose_xyz_left = observations["robot__actual__poses__right::panda__xyz"]
        pose_xyz_right = observations["robot__actual__poses__left::panda__xyz"]
        gripper_state_left = observations["robot__actual__grippers__right::panda_hand"]
        gripper_state_right = observations["robot__actual__grippers__left::panda_hand"]

        # We only use the scene_right_0 camera, because this is the camera that I moved to be an egocentric viewpoint 
        for camera_name in ["scene_right_0"]:
            camera_id = camera_names[camera_name]

            print(f"Loading {camera_name} images...")
            images = observations[camera_id]
            print(f"Loading intrisnics and extrinsics...")
            intrinsics = np.load(os.path.join(episode_dir, "processed", "intrinsics.npz"))[camera_id]
            extrinsics = np.load(os.path.join(episode_dir, "processed", "extrinsics.npz"))[camera_id]
            print(f"Done loading.")

            left_pixel_points = np.array([project_3d_to_pixels(pose_xyz_left[i], intrinsics, extrinsics[i]) for i in range(pose_xyz_left.shape[0])])
            right_pixel_points = np.array([project_3d_to_pixels(pose_xyz_right[i], intrinsics, extrinsics[i]) for i in range(pose_xyz_right.shape[0])])
            
            left_gripper_closed = gripper_state_left < 0.08
            right_gripper_closed = gripper_state_right < 0.08 

            initial_frame = deepcopy(images[0])

            # @Zeqing this part shows how to project the gripper positions into the coordinate frame of the camera at t=0
            # Note that the camera doesn't move in this data, so it doesn't make a difference, but if the camera did move, this is how to do it. 
            # I would still add this logic to the MotionTrans data loader though, so that if we end up using robot data in the future where the camera
            # does move, we have the proper code in place. 
            extrinsics_t0 =  extrinsics[0]
            intrinsics_t0 =  intrinsics
            pose_xyz_left_t0 = np.array([project_3d_to_3d(pose_xyz_left[i], extrinsics_t0) for i in range(pose_xyz_left.shape[0])])
            pose_xyz_right_t0 = np.array([project_3d_to_3d(pose_xyz_right[i], extrinsics_t0) for i in range(pose_xyz_right.shape[0])])

            # Project to the 2D camera plane at t=0 for visualization 
            left_pixel_points_t0 = np.array([project_from_3d_to_pixels(pose_xyz_left_t0[i], intrinsics_t0) for i in range(pose_xyz_left_t0.shape[0])])
            right_pixel_points_t0 = np.array([project_from_3d_to_pixels(pose_xyz_right_t0[i], intrinsics_t0) for i in range(pose_xyz_right_t0.shape[0])])

            
            for i, frame in enumerate(images):

                RED = (255, 0, 0)
                BLUE = (0, 0, 255)
                GREEN = (0, 255, 0)
                WHITE = (255, 255, 255)

                for handside in ["left", "right"]:
                    color = RED if handside == "left" else BLUE

                    pt = left_pixel_points[i] if handside == "left" else right_pixel_points[i]
                    frame = cv2.circle(frame, pt, 5, color, -1)  # filled circle
                    frame = cv2.putText(frame, handside, pt, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # Resize to whatever size you need for your data processing
                    frame = cv2.resize(frame, (640, 480)) 

                    new_frames.append(frame)

                    pt_t0 = left_pixel_points_t0[i] if handside == "left" else right_pixel_points_t0[i]
                    initial_frame = cv2.circle(initial_frame, pt_t0, 5, color, -1)  # filled circle
                    # initial_frame = cv2.putText(initial_frame, handside, pt_0, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    FPS = 10
    video_file = os.path.join(episode_dir, "annotated", "scene_right_0.mp4")
    os.makedirs(os.path.dirname(video_file), exist_ok=True)
    imageio.mimsave(video_file, new_frames, fps=FPS, macro_block_size=1)
    print(f"Saved episode images to {video_file}.")
    
    initial_frame_file = os.path.join(episode_dir, "annotated", "initial_frame_annotated.png")
    imageio.imsave(initial_frame_file, initial_frame)
    print(f"Saved initial frame to {initial_frame_file}.")
                
"""
Example usage: 

python3 -u load_lbm_sim.py \
--episode-dir "/data/kylehatch/LBM_sim_egocentric/PutBananaOnSaucer/cabot/sim/bc/teleop/2024-11-14T14-54-50-08-00/diffusion_spartan/episode_0"


"""