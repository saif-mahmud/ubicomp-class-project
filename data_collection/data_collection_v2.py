import argparse
import json
import os
import re
import time
from datetime import datetime

import cv2
import serial
import yaml
from playsound import playsound
import numpy as np

from utils.commands import generate_discrete

karaoke = False  # Flag for karaoke effects

def load_config(config_path):
    if os.path.exists(config_path):
        config = json.load(open(config_path, 'rt'))
    else:
        config = json.load(open('./configs/demo_config_nRF52840.json', 'rt'))
        config['audio']['files'] = []
        config['audio']['syncing_poses'] = []
        config['ground_truth']['files'] = []
        config['ground_truth']['videos'] = []
        config['ground_truth']['syncing_poses'] = []
        config['sessions'] = []
    return config

def get_serial_port():
    all_dev = os.listdir('/dev')
    serial_ports = ['/dev/' + x for x in all_dev if x[:6] == 'cu.usb']
    if len(serial_ports) == 0:
        manual_serial_port = input(
            'No serial port found, please specify serial port name: ')
        serial_ports += [manual_serial_port.rstrip('\n')]
    selected = 0
    if len(serial_ports) > 1:
        print('Multiple serial ports found, choose which one to use (0-%d)' %
              (len(serial_ports) - 1))
        for n, p in enumerate(serial_ports):
            print('%d: /dev/%s' % (n, p))
        selected = int(input())
    return serial_ports[selected]

def load_label(exc_cfg, mode, folds, reps_per_fold, shuffle=False):
    cmds = {'idx': exc_cfg[mode]['idx'],
            'cmds': exc_cfg[mode]['activities'],
            'durations': exc_cfg[mode]['durations'],
            'instruction_imgs': exc_cfg[mode]['instruction_imgs'],
            'audio_instrctions': exc_cfg[mode]['audio_instrctions'],
            'video_instructions': exc_cfg[mode]['video_instructions']}  # Add video instructions

    cmd_list, _, _ = generate_discrete(
        cmds=cmds, folds=folds, reps_per_fold=reps_per_fold, shuffle=shuffle)

    audio_dict = dict(zip(cmds['cmds'], cmds['audio_instrctions'])) if len(
        cmds['cmds']) == len(cmds['audio_instrctions']) else {}

    video_dict = dict(zip(cmds['cmds'], cmds['video_instructions'])) if len(
        cmds['cmds']) == len(cmds['video_instructions']) else {}

    return cmd_list, audio_dict, video_dict  # Return video_dict

def play_video(video_path, main_window):
    """Play video and overlay it on the main window."""
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size = (320, 240)  # Adjust size as needed

    while video_cap.isOpened():
        ret, video_frame = video_cap.read()
        if not ret:
            break

        video_frame = cv2.resize(video_frame, video_size)
        x_offset = 10  # Horizontal position
        y_offset = 10  # Vertical position
        main_window[y_offset:y_offset + video_size[1], x_offset:x_offset + video_size[0]] = video_frame

        cv2.imshow('Activity Recognition / Pose Estimation - Data Collection', main_window)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit
            break

    video_cap.release()

def add_padding(frame, target_width, target_height):
    """
    Resize the frame while maintaining aspect ratio and add black padding to fit the target size.
    For vertical videos (height > width), padding is added to the sides.
    """
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    # Calculate new dimensions to maintain aspect ratio
    if aspect_ratio < (target_width / target_height):
        # Fit to height (vertical video)
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    else:
        # Fit to width (horizontal video)
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Mirror the frame horizontally
    resized_frame = cv2.flip(resized_frame, 1)

    # Create a black background
    padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate padding offsets
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized frame in the center of the black background
    padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

    return padded_frame

def data_record(path_prefix, output_path, cmd_set, duration, folds, n_reps_per_fold, noserial, count_down,
                play_audio_instruction: bool):
    if not os.path.exists(os.path.join(path_prefix, output_path)):
        print('Creating path :', os.path.join(path_prefix, output_path))
        os.makedirs(os.path.join(path_prefix, output_path))

    config_path = os.path.join(path_prefix, output_path, 'config.json')
    config = load_config(config_path)

    exercise_config_file = open("./configs/exercise_config.yaml", mode="r")
    exc_cfg = yaml.load(exercise_config_file, Loader=yaml.FullLoader)

    cap = cv2.VideoCapture(0)
    frame_size = (1280, 720)  # (width, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = 20  # Fallback FPS if not detected

    save_pos = datetime.now().strftime('record_%Y%m%d_%H%M%S_%f')
    vid = cv2.VideoWriter(os.path.join(path_prefix, output_path, save_pos + '.mp4'),
                          cv2.VideoWriter_fourcc(*"mp4v"), int(actual_fps), frame_size)

    cmds, audio_instruction_dict, video_instruction_dict = load_label(exc_cfg=exc_cfg, mode=cmd_set, folds=folds,
                                                                      reps_per_fold=n_reps_per_fold, shuffle=True)

    audio_filename = None

    if not noserial:
        serial_port = get_serial_port()
        ser = serial.Serial(serial_port, 115200)
        print('Listening to Serial Port :', serial_port)
        ser.write(b's')
        print('Start signal sent')

        start_notice = ser.readline()

        if len(start_notice):
            filename_match = re.findall(
                r'(audio\d+\.raw)', start_notice.decode())
            if len(filename_match):
                audio_filename = filename_match[0]
                print('Audio filename: %s' % audio_filename)

    cmd_idx = 0
    last_cmd_display_time = 0

    exit_flag = False
    syncing_received = False

    t0 = 0
    n_frames = 0

    ts, rcds = [], []

    video_frames = []  # Preloaded frames for current instruction
    video_start_time = 0

    try:
        session_idx = len(config['ground_truth']['files']) + 1
        print(f"Session # {session_idx}")

        while cap.isOpened():
            if t0 == 0:
                t0 = time.time()

            success, raw_frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            n_frames += 1
            frame_ts = time.time()

            if not noserial and ser.inWaiting():
                in_bytes = ser.readline()
                if in_bytes == b'000\n':
                    syncing_received = True
                    syncing_frame = n_frames
                    syncing_ts = frame_ts
                    print('Received syncing signal, frame # %d, ts %.3f' %
                          (syncing_frame, syncing_ts))

            time_from_start = time.time() - t0
            info_text = f'Frame # {n_frames:06d}, timestamp: {frame_ts:0.6f}'

            image = cv2.flip(raw_frame, 1)
            cv2.putText(image, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)

            vid.write(image)

            if time_from_start >= count_down and not exit_flag:
                info_text += f', {cmds[cmd_idx][1]}'

                if cmds[cmd_idx][1] in video_instruction_dict:
                    video_path = video_instruction_dict[cmds[cmd_idx][1]]
                    if not video_frames:
                        video_cap = cv2.VideoCapture(video_path)
                        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                        video_start_time = time.time()
                        while True:
                            ret, frame = video_cap.read()
                            if not ret:
                                break
                            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                            frame = cv2.flip(frame, 1)
                            padded_frame = add_padding(frame, 640, 720)
                            video_frames.append(padded_frame)
                        video_cap.release()

                    if video_frames:
                        elapsed_time = time.time() - video_start_time
                        frame_idx = int(elapsed_time * video_fps) % len(video_frames)
                        instruction_frame = video_frames[frame_idx]
                        image[0:720, 0:640] = instruction_frame

            ts += [frame_ts]

            if not noserial and not syncing_received:
                print('Waiting for Teensy...')
                time.sleep(0.05)
                continue

            if time_from_start < count_down:
                cv2.putText(image, 'Please Clap', (300, 380), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255),
                            thickness=4)
                cv2.putText(image,
                            f"Session {session_idx} starting in {int(count_down - time_from_start)} seconds",
                            (250, 480),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), thickness=2)

            elif not exit_flag:
                # Reset only the right half of the image (where text, progress bar, and camera feed will be)
                image[:, 640:] = 0  # Reset everything on the right half

                if len(cmds[cmd_idx]) == 4:
                    duration = cmds[cmd_idx][3]

                # Display info text on the top right half
                cv2.putText(image, info_text, (650, 30),  # Position text on the top right
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)

                estimated_time_left = 0

                for i in range(cmd_idx, len(cmds)):
                    if len(cmds[cmd_idx]) == 4:
                        estimated_time_left += cmds[i][3]
                    else:
                        estimated_time_left += duration

                if time.time() - last_cmd_display_time > duration:
                    if last_cmd_display_time > 0 and not bad_sample:
                        if cmds[cmd_idx][0] >= 0:
                            rcds += [(str(cmds[cmd_idx][0]), cmd_start_time,
                                    time.time(), cmds[cmd_idx][1])]
                        cmd_idx += 1

                        video_frames = []
                        video_start_time = 0

                    if play_audio_instruction and len(audio_instruction_dict) != 0:
                        playsound(audio_instruction_dict[cmds[cmd_idx][1]], block=False)

                    last_cmd_display_time = time.time()
                    cmd_start_time = time.time()
                    bad_sample = False

                elif last_cmd_display_time > 0:
                    cmd_progress = (
                        time.time() - last_cmd_display_time) / duration

                    # Progress bar on the top right half
                    image[340:350, 640: round(
                        cmd_progress * 640) + 640] = 255

                    cmd_text = ' '.join(cmds[cmd_idx][1].split()[:3])

                    # Display command text on the top right half
                    text_size = cv2.getTextSize(
                        '%s' % cmd_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0][0]
                    cv2.putText(image, '%s' % cmd_text, (650, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=4)

                # Display session progress on the top right half (above the progress bar and command text)
                progress_text = f'Session progress: {(cmd_idx / len(cmds) * 100):0.1f}%, estimated time left: {estimated_time_left} s'
                text_size = cv2.getTextSize(
                    progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
                cv2.putText(image, progress_text, (650, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), thickness=2)
                
                camera_frame = raw_frame.copy()
                camera_frame = cv2.resize(camera_frame, (640, 360))
                camera_frame = cv2.flip(camera_frame, 1)
                image[360:720, 640:1280] = camera_frame

            cv2.imshow(
                'Activity Recognition / Pose Estimation - Data Collection', image)
            pressed_key = cv2.waitKeyEx(1) & 0xFF

            if not exit_flag:
                if cmd_idx == len(cmds):
                    session_end_time = time.time()
                    exit_flag = True
                    break
                if pressed_key == 27:
                    session_end_time = time.time()
                    exit_flag = True
                    break
                if pressed_key == ord('x') or pressed_key == 2:
                    bad_sample = True
                    last_cmd_display_time = time.time() - duration
                    print('Bad:', frame_ts)
                if (pressed_key == ord(' ') or pressed_key == 3) and time.time() - last_cmd_display_time > 0.6:
                    last_cmd_display_time = min(
                        last_cmd_display_time, time.time() - duration + 0.3)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if not noserial:
            ser.write(b'e')

        cap.release()
        vid.release()
        cv2.destroyAllWindows()

    cap.release()
    vid.release()
    cv2.destroyAllWindows()
    with open(os.path.join(path_prefix, output_path, save_pos + '_records.txt'), 'wt') as f:
        for r in rcds:
            f.write('%s,%f,%f,%s\n' % r)

    # updating config file here
    if audio_filename:
        config['audio']['files'] += [audio_filename]

    # ground truth file format: classification - .txt , regression - .npy
    config['ground_truth']['files'] += [
        os.path.basename(os.path.join(path_prefix, output_path, save_pos + '_records.txt'))]
    config['ground_truth']['videos'] += [os.path.basename(
        os.path.join(path_prefix, output_path, save_pos + '.mp4'))]
    config['sessions'] += [[]]

    for fold_start_pos in range(0, len(rcds), len(cmds) // folds):
        fold_end_pos = min(fold_start_pos + len(cmds) // folds, len(rcds)) - 1
        fold_start_ts = rcds[fold_start_pos][1]
        fold_duration = rcds[fold_end_pos][2] + 0.01 - fold_start_ts
        config['sessions'][-1] += [{
            'start': fold_start_ts,
            'duration': fold_duration
        }]

    json.dump(config, open(config_path, 'wt'), indent=4)

    with open(os.path.join(path_prefix, output_path, save_pos + '_frame_time.txt'), 'wt') as f:
        for t in ts:
            f.write('%f\n' % (t))

    if not noserial:
        print(ser.readline().decode())
        if ser.inWaiting():
            print(ser.readline().decode())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path-prefix', help='datareader parent folder',
                        default='/Users/zrd/research_projects/echowrist/pilot_study')
    parser.add_argument('-o', '--output', help='output dir name')
    parser.add_argument('-c', '--commandsets',
                        help='command set name, comma separated if multiple', default='15')
    parser.add_argument(
        '-t', '--time', help='duration of each command/gesture', type=float, default=3)
    parser.add_argument(
        '-f', '--folds', help='how many folds', type=int, default=1)
    parser.add_argument('-r', '--reps_per_fold', help='how many repetitiions per gesture for a fold', type=int,
                        default=1)
    parser.add_argument('-cd', '--count_down',
                        help='count down time (s) before start', type=int, default=3)
    parser.add_argument(
        '--noserial', help='do not listen on serial', action='store_true')
    parser.add_argument(
        '--play_audio', help='play command audio', action='store_true')

    args = parser.parse_args()

    data_record(args.path_prefix, args.output, args.commandsets, args.time, args.folds, args.reps_per_fold,
                args.noserial, args.count_down, args.play_audio)