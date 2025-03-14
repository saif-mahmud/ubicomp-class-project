import argparse
import torch
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_frame_time
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from hmr2.utils.renderer import Renderer
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from detectron2 import model_zoo

import warnings
warnings.filterwarnings("ignore")

# Define a light blue color for rendering
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def plot_skeleton_3D(ax, skeleton_coords, link_color="blue", joint_color="green"):
    """
    Plot a 3D skeleton using the given coordinates.
    Args:
        ax: Matplotlib 3D axis object.
        skeleton_coords: 3D coordinates of the skeleton joints.
        link_color: Color of the links between joints.
        joint_color: Color of the joints.
    """
    # Define the links between joints
    links_list = [
        [0, 15, 17],  # Right leg
        [0, 16, 18],  # Left leg
        [0, 1, 8],    # Spine
        [1, 2, 3, 4], # Right arm
        [1, 5, 6, 7], # Left arm
        [8, 9, 10, 11, 22], # Right side of the body
        [8, 12, 13, 14, 19], # Left side of the body
    ]
    # Plot joints
    ax.scatter3D(skeleton_coords[:, 0], skeleton_coords[:, 2], -skeleton_coords[:, 1], color=joint_color)
    # Plot links between joints
    for links in links_list:
        ax.plot3D(
            xs=skeleton_coords[links, 0],
            ys=skeleton_coords[links, 2],
            zs=-skeleton_coords[links, 1],
            color=link_color
        )
    # Set axis limits
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)


def body_landmarks_hmr(source_video_path, timestamp_file_path, target_video_path, ground_truth_file_path, hmr_gpu=-1, detectron_gpu=-1, arm_length=60, batch_size=32):
    """
    Process a video to extract 3D body landmarks using the HMR2 model.
    Args:
        source_video_path: Path to the input video file.
        timestamp_file_path: Path to the file containing frame timestamps.
        target_video_path: Path to save the output video with rendered poses.
        ground_truth_file_path: Path to save extracted 3D pose data.
        hmr_gpu: GPU index for the HMR2 model (-1 for CPU).
        detectron_gpu: GPU index for the Detectron2 model (-1 for CPU).
        arm_length: Length of the arm for normalization (not used in this implementation).
        batch_size: Number of frames to process in a batch.
    """
    # Load the HMR2 model and its configuration
    hmr_model, hmr_model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    # Set the device (CPU or GPU) for the HMR2 model
    device = torch.device('cpu' if hmr_gpu == -1 or not torch.cuda.is_available() else f'cuda:{hmr_gpu}')
    print(f'Device for HMR2.0 model: {device}')
    hmr_model = hmr_model.to(device)
    hmr_model.eval()

    # Load the Detectron2 configuration and set up the detector
    detectron2_cfg = model_zoo.get_config(
        'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg, gpu_idx=detectron_gpu)

    # Load frame timestamps
    ts = load_frame_time(timestamp_file_path)

    # Open the video file
    cap = cv2.VideoCapture(source_video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'\n[SMPL Ground Truth] Src Path: {source_video_path} | Total Frames: {num_frames}')

    # Initialize video writer and progress bar
    vid = None
    pbar = tqdm(total=num_frames)
    save_data = []
    current_frame_index = 0

    # Process frames in batches
    while True:
        frames = []
        frame_indices = []
        for _ in range(batch_size):
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
            frame_indices.append(current_frame_index)  # Track frame indices for timestamp lookup
            current_frame_index += 1
            pbar.update(1)

        if not frames:
            break  # Exit if no more frames are available

        # Detect humans in all frames in the batch
        det_outs = [detector(frame) for frame in frames]
        det_instances_list = [det_out['instances'] for det_out in det_outs]
        valid_idx_list = [(det_instances.pred_classes == 0) & (det_instances.scores > 0.5) for det_instances in det_instances_list]
        boxes_list = [det_instances.pred_boxes.tensor[valid_idx].cpu().numpy() for det_instances, valid_idx in zip(det_instances_list, valid_idx_list)]

        # Initialize the renderer
        renderer = Renderer(hmr_model_cfg, faces=hmr_model.smpl.faces)

        # Process each frame in the batch
        for i, (frame, boxes) in enumerate(zip(frames, boxes_list)):
            if len(boxes) == 0:
                continue  # Skip frames with no detected humans

            # Create a dataset for the current frame and detected boxes
            dataset = ViTDetDataset(hmr_model_cfg, frame, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = hmr_model(batch)  # Run the HMR2 model

                # Render the mesh on the image patch
                regression_img = renderer(
                    out['pred_vertices'][0].detach().cpu().numpy(),
                    out['pred_cam_t'][0].detach().cpu().numpy(),
                    batch['img'][0],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )
                regression_img = cv2.cvtColor(regression_img, cv2.COLOR_RGB2BGR)

                # Place the rendered mesh back on the original frame
                smpl_img = frame.copy()
                box_width = regression_img.shape[1]
                box_height = regression_img.shape[0]
                box_center_np = batch["box_center"].float().detach().cpu().numpy()[0]
                box_center_x, box_center_y = int(box_center_np[0]), int(box_center_np[1])
                scaled_width = int(box_width * batch['downsampling_factor'])
                scaled_height = int(box_height * batch['downsampling_factor'])
                try:
                    smpl_img[
                        box_center_y - scaled_height // 2: box_center_y + scaled_height // 2,
                        box_center_x - scaled_width // 2: box_center_x + scaled_width // 2
                    ] = cv2.resize(regression_img.copy(), (scaled_width // 2 * 2, scaled_height // 2 * 2)) * 255
                except:
                    continue  # Skip if the mesh is out of frame

                # Extract 3D keypoints
                coords = out['pred_keypoints_3d'][0].cpu().reshape(1, -1, 3)[0][0:25,]

                # Plot the 3D skeleton
                fig = plt.figure(1, figsize=plt.figaspect(1))
                fig.clf()
                ax1 = fig.add_subplot(1, 1, 1, projection='3d')
                ax1.view_init(5, -55)
                plot_skeleton_3D(ax1, coords)
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                plt_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                plt_img = plt_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
                plt.close()

                # Assemble the final frame (original + mesh + skeleton)
                skeleton_img = np.zeros((frame.shape[0], plt_img.shape[1], frame.shape[2]), np.uint8)
                skeleton_img[0:plt_img.shape[0], :, :] = plt_img
                skeleton_img = cv2.cvtColor(skeleton_img, cv2.COLOR_RGB2BGR)
                assembled_frame = np.concatenate([frame, smpl_img, skeleton_img], axis=1)

                # Initialize the video writer if not already done
                if vid is None:
                    vid = cv2.VideoWriter(
                        target_video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30,
                        (assembled_frame.shape[1], assembled_frame.shape[0])
                    )
                vid.write(np.uint8(assembled_frame))

                # Save frame data (timestamps, keypoints, SMPL parameters)
                frame_data = np.zeros((302,))
                frame_data[0] = ts[frame_indices[i]]  # Timestamp
                frame_data[1:76] = coords.reshape(1, 25 * 3)  # 3D keypoints [75]
                frame_data[76:85] = out["pred_smpl_params"]['global_orient'][0].cpu().reshape(1, 3 * 3)  # Global orientation [9]
                frame_data[85:292] = out["pred_smpl_params"]['body_pose'][0].cpu().reshape(1, 23 * 3 * 3)  # Body pose [207]
                frame_data[292:302] = out["pred_smpl_params"]['betas'][0].cpu().reshape(1, 10)  # Shape parameters [10]
                save_data.append(frame_data)

    # Release resources
    cap.release()
    if vid is not None:
        vid.release()
    save_data = np.array(save_data)
    np.save(ground_truth_file_path, save_data)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='Path to the video file')
    parser.add_argument('-l', '--arm-length', default=60, type=int, help='Arm length for normalization')
    parser.add_argument('-hg', '--hmr-gpu', type=int, default=-1, help='GPU index for HMR model')
    parser.add_argument('-dg', '--detectron-gpu', type=int, default=-1, help='GPU index for Detectron model')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size for processing frames')

    args = parser.parse_args()

    start = time.perf_counter() 
    body_landmarks_hmr(args.video, 
                       f'{args.video[:-4]}_frame_time.txt',
                       f'{args.video[:-4]}_hmr2_pose_video.mp4', 
                       f'{args.video[:-4]}_hmr2_pose_landmarks.npy', 
                       args.hmr_gpu, args.detectron_gpu, args.arm_length, args.batch_size)
    end = time.perf_counter()

    elapsed = end - start
    print(f"\nElapsed: {elapsed:.4f} seconds")