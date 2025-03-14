import argparse
import os

import cv2
import matplotlib.animation as animation
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from plot_utils import time_animate
from utils import load_frame_time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def detect_pose_single_frame(image, pose_detector):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose_detector.process(image)

    if not results.pose_landmarks:
        return None, None

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(image,
                              results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    return image, results


def mediapipe_pose_estimation(source_video_path, target_video_path, timestamp_file_path, ground_truth_file_path,
                              visualize=False):
    frame_time = load_frame_time(timestamp_file_path)

    mp_pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(source_video_path)

    success, frame = cap.read()
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height, frame_width, _ = frame.shape

    print(
        f'[MEDIAPIPE PROC] Src Path : {source_video_path} | Input Video Shape: {frame.shape} | Total Frames : {num_frames}')

    vid = cv2.VideoWriter(target_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                          (frame_width, frame_height))

    pbar = tqdm(total=num_frames)
    data = list()
    frame_cnt = 0
    err_log = 0

    while success:
        try:
            _frame_data = [frame_time[frame_cnt]]
            annotated_frame, mp_output = detect_pose_single_frame(frame, mp_pose_detector)

            success, frame = cap.read()

            frame_cnt += 1
            pbar.update(1)

            if mp_output is None:
                continue

            landmarks = mp_output.pose_world_landmarks.landmark

            for i in range(len(mp_pose.PoseLandmark)):
                _frame_data.extend([landmarks[i].x, landmarks[i].y, landmarks[i].z])

            data.append(_frame_data)
            vid.write(annotated_frame)

        except Exception as err:
            err_log += 1
            # traceback.print_exc()
            print(str(err))
            continue

    pbar.close()
    cap.release()
    vid.release()

    data = np.array(data)
    np.save(ground_truth_file_path, data)

    if visualize:
        anim = time_animate(np.transpose(data), mp_pose.POSE_CONNECTIONS)

        writervideo = animation.FFMpegWriter(fps=30)
        anim.save(f'{source_video_path[:-4]}_pose_vis.mp4', writer=writervideo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pose Ground Truth Using Mediapipe')
    parser.add_argument('-p', '--path', help='path to the input video file .mp4')
    args = parser.parse_args()
    # data_dir = '../data/pilot_study/P01/session_0103/datareader/session_0101'
    mediapipe_pose_estimation(source_video_path=os.path.join(args.path, 'reference_video.mp4'),
                              target_video_path=os.path.join(args.path, 'ground_truth_pose_video.mp4'),
                              ground_truth_file_path=os.path.join(args.path, 'pose_ground_truth.npy'),
                              timestamp_file_path=os.path.join(args.path, 'reference_video_frame_time.txt'))
