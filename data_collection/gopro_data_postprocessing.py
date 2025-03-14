import argparse
import csv
import datetime
import glob
import json
import os.path
import subprocess
import time

import cv2

from data_collection import load_config


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def get_video_timestamp(video_filepath: str):
    # subprocess to extract creation time from metadata
    command = f'ffprobe -v quiet -select_streams v:0  -show_entries stream_tags=creation_time -of default=noprint_wrappers=1:nokey=1 {video_filepath}'
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=None, shell=True)
    output = process.communicate()[0]
    output = output.decode("utf-8")

    year = int(output[:4])
    month = int(output[5:7])
    date = int(output[8:10])
    hour = int(output[11:13])
    minute = int(output[14:16])
    second = int(output[17:19])  # takes rounded int value for seconds

    t = datetime.datetime(year, month, date, hour, minute, second)

    print(f'[DEBUG] retrieved timestamp: {t}')
    # gmt = pytz.timezone('GMT')
    # eastern = pytz.timezone('US/Eastern')
    # t = gmt.localize(t)
    # t = t.astimezone(eastern)

    creation_time_unix = float(time.mktime(t.timetuple()))

    updated_fname = t.strftime('record_%Y%m%d_%H%M%S_%f')

    return creation_time_unix, updated_fname


def get_frame_times(video_filepath, initial_unix_time, updated_fname, count_down):
    print('[INFO] Compressing ref video')

    updated_vid_fname = os.path.join(os.path.dirname(
        video_filepath), updated_fname + '.mp4')
    command = f'ffmpeg -y -i {video_filepath} -s 640x480 -vcodec libx264 -crf 30 -preset ultrafast {updated_vid_fname}'
    subprocess.run(command, shell=True)
    # shutil.copyfile(video_filepath, updated_vid_fname)

    tmp_file = './temp.csv'
    command = f'ffprobe -select_streams v -show_entries packet=pts_time -of csv {video_filepath} > {tmp_file}'
    subprocess.run(command, shell=True)

    increments = []

    with open(tmp_file) as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            increments.append(row[1])

    print('[INFO] Creating clap video')
    clap_vid_fname = os.path.join(os.path.dirname(
        video_filepath), 'clap_' + updated_fname + '.mp4')

    cap = cv2.VideoCapture(video_filepath)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)

    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'[DEBUG] fps: {fps}')

    clap_frames = int(fps * (count_down + fps))

    vid = cv2.VideoWriter(
        clap_vid_fname, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    with open(os.path.join(os.path.dirname(video_filepath), updated_fname + '_frame_time.txt'), 'w') as f:
        frame_no = 0
        while total_frames >= frame_no + 1:
            # writing to frame_txt file
            frame_ts = initial_unix_time + float(increments[frame_no])
            f.write(f'{frame_ts:0.6f}\n')

            if frame_no <= clap_frames:
                frame_exists, curr_frame = cap.read()

                if frame_exists:
                    info_text = f'Frame # {frame_no:06d}, timestamp: {frame_ts:0.6f}'

                    cv2.putText(curr_frame, info_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), thickness=2)
                    vid.write(curr_frame)

            frame_no += 1

    vid.release()
    cap.release()
    os.remove(tmp_file)
    print(f'[INFO] Timestamped and compressed video(s) saved')


def update_config(data_dir, updated_fname, init_unix_ts, vid_len, count_down_time, folds):
    config_path = os.path.join(data_dir, 'config.json')
    config = load_config(config_path)

    config['ground_truth']['videos'] += [(updated_fname + '.mp4')]
    config['sessions'] += [[]]

    fold_start_ts = init_unix_ts + count_down_time
    hop_len, last_hop = vid_len / folds, vid_len % folds

    for i in range(folds):
        fold_duration = hop_len if i != (folds - 1) else (hop_len + last_hop)

        config['sessions'][-1] += [{
            'start': fold_start_ts,
            'duration': fold_duration
        }]

        fold_start_ts = fold_start_ts + hop_len

    json.dump(config, open(config_path, 'wt'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--path', help='File path of the data collection directory containing video(s)')
    parser.add_argument('-cd', '--count_down',
                        help='count down time (s) before start', type=int, default=15)
    parser.add_argument(
        '-fc', '--force', help='force overwrite echo profile files', action='store_true')
    parser.add_argument(
        '-f', '--folds', help='how many folds', type=int, default=1)

    args = parser.parse_args()

    # avi_files = glob.glob(os.path.join(args.path, '*.avi'))

    # print('[INFO] converting to mp4')
    # for i, avi in enumerate(avi_files): 
    #     convert_command = f'ffmpeg -i {avi} -strict -2 {os.path.join(args.path, "output" + str(i) + ".MP4")}'
    #     subprocess.run(convert_command, shell=True)

    vid_files = glob.glob(os.path.join(args.path, '*.MP4'))

    print(f'[DEBUG] video file(s): {vid_files}')

    start = time.time()

    for ref_vid_path in vid_files:
        if ("record" not in ref_vid_path) or args.force:
            print('[INFO] Getting frame timestamps')
            print('[INFO] working on' + ref_vid_path)
            initial_unix_time, updated_fname = get_video_timestamp(
                ref_vid_path)

            get_frame_times(ref_vid_path, initial_unix_time,
                            updated_fname, args.count_down)

            vid_len = get_length(ref_vid_path)

            print('[INFO] Creating / updating config file')
            update_config(args.path, updated_fname, initial_unix_time,
                          vid_len, args.count_down, args.folds)

            # removing the original gopro video (very large in size)
            # print(f'[INFO] Deleting original video')
            # os.remove(ref_vid_path)

    print(f'elapsed time: {time.time() - start} sec')
