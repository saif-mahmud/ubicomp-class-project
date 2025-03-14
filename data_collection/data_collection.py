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

from utils.commands import generate_discrete

karaoke = False  # Flag for karaoke effects


def load_config(config_path):
    if os.path.exists(config_path):
        config = json.load(open(config_path, 'rt'))
    else:
        config = json.load(open('./configs/demo_config_teensy41.json', 'rt'))
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
            'audio_instrctions': exc_cfg[mode]['audio_instrctions']}

    cmd_list, _, _ = generate_discrete(
        cmds=cmds, folds=folds, reps_per_fold=reps_per_fold, shuffle=shuffle)

    audio_dict = dict(zip(cmds['cmds'], cmds['audio_instrctions'])) if len(
        cmds['cmds']) == len(cmds['audio_instrctions']) else {}

    return cmd_list, audio_dict


def data_record(path_prefix, output_path, cmd_set, duration, folds, n_reps_per_fold, noserial, count_down,
                play_audio_instruction: bool):
    # create directory for the data
    if not os.path.exists(os.path.join(path_prefix, output_path)):
        print('Creating path :', os.path.join(path_prefix, output_path))
        os.makedirs(os.path.join(path_prefix, output_path))

    # load template config file
    config_path = os.path.join(path_prefix, output_path, 'config.json')
    config = load_config(config_path)

    # load exercise / activity config
    exercise_config_file = open("./configs/exercise_config.yaml", mode="r")
    exc_cfg = yaml.load(exercise_config_file, Loader=yaml.FullLoader)

    # ground truth video recording (0 = webcam)
    cap = cv2.VideoCapture(0)

    frame_size = (1280, 720)  # (width, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    save_pos = datetime.now().strftime('record_%Y%m%d_%H%M%S_%f')
    vid = cv2.VideoWriter(os.path.join(path_prefix, output_path, save_pos + '.mp4'),
                          cv2.VideoWriter_fourcc(*"mp4v"), 30, frame_size)

    # loading set of activity classes for data recording
    cmds, audio_instruction_dict = load_label(exc_cfg=exc_cfg, mode=cmd_set, folds=folds,
                                              reps_per_fold=n_reps_per_fold, shuffle=True)

    audio_filename = None

    # listening to serial port for connecting to microcontroller (i.e. teensy)
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

    try:
        session_idx = len(config['ground_truth']['files']) + 1
        print(f"Session # {session_idx}")

        while cap.isOpened():
            if t0 == 0:
                t0 = time.time()

            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            n_frames += 1
            frame_ts = time.time()

            if not noserial and ser.inWaiting():
                in_bytes = ser.readline()
                # while in_bytes != b'000\n': pass
                if in_bytes == b'000\n':
                    syncing_received = True
                    syncing_frame = n_frames
                    syncing_ts = frame_ts
                    print('Received syncing signal, frame # %d, ts %.3f' %
                          (syncing_frame, syncing_ts))

            image = cv2.flip(image, 1)

            time_from_start = time.time() - t0
            info_text = f'Frame # {n_frames:06d}, timestamp: {frame_ts:0.6f}'

            if time_from_start >= count_down and not exit_flag:
                info_text += f', {cmds[cmd_idx][1]}'

            cv2.putText(image, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
            vid.write(image)

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
                # image[:] = 0

                # print(cmds, '\n\n')
                if len(cmds[cmd_idx]) == 4:
                    duration = cmds[cmd_idx][3]

                cv2.putText(image, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
                # image[:] = 255
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

                    if play_audio_instruction and len(audio_instruction_dict) != 0:
                        playsound(
                            audio_instruction_dict[cmds[cmd_idx][1]], block=False)

                    last_cmd_display_time = time.time()
                    cmd_start_time = time.time()
                    bad_sample = False

                elif last_cmd_display_time > 0:
                    cmd_progress = (
                        time.time() - last_cmd_display_time) / duration

                    image[470:472, 0: round(
                        cmd_progress * image.shape[1])] = 255

                    cmd_text = ' '.join(cmds[cmd_idx][1].split()[:3])

                    text_size = cv2.getTextSize(
                        '%s' % cmd_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)[0][0]
                    cv2.putText(image, '%s' % cmd_text, ((1280 - text_size) // 2, 350), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                (0, 0, 255), thickness=8)

                progress_text = f'Session progress: {(cmd_idx / len(cmds) * 100):0.1f}%, estimated time left: {estimated_time_left} s'
                text_size = cv2.getTextSize(
                    progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
                cv2.putText(image, progress_text, ((1280 - text_size) // 2, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), thickness=2)
            else:
                break

            cv2.imshow(
                'Activity Recognition / Pose Estimation - Data Collection', image)
            pressed_key = cv2.waitKeyEx(1) & 0xFF

            if not exit_flag:
                if cmd_idx == len(cmds):
                    session_end_time = time.time()
                    exit_flag = True
                if pressed_key == 27:
                    session_end_time = time.time()
                    exit_flag = True
                    # break
                if pressed_key == ord('x') or pressed_key == 2:
                    bad_sample = True
                    last_cmd_display_time = time.time() - duration
                    print('Bad:', frame_ts)
                if (pressed_key == ord(' ') or pressed_key == 3) and time.time() - last_cmd_display_time > 0.6:
                    last_cmd_display_time = min(
                        last_cmd_display_time, time.time() - duration + 0.3)

    except:
        pass

    if not noserial:
        ser.write(b'e')

    cap.release()
    vid.release()

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
