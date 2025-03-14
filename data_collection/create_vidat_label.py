import argparse
import csv
import glob
import json
import os

import cv2

from data_collection import load_config


def create_labels(data_path, updated_fname, annotation_file, fps, config):
    vidat_annotation = os.path.join(data_path, annotation_file)
    frame_timestamps = os.path.join(
        data_path, updated_fname + '_frame_time.txt')
    out_file = os.path.join(data_path, updated_fname + '_records.txt')

    frame_to_timestamp_dict = dict()
    # Start by populating the frame-to-timestamp dictionary based on the text file
    with open(frame_timestamps) as f_to_t:
        frame_number = 1
        for line in f_to_t:
            frame_to_timestamp_dict[frame_number] = line.strip("\n")
            frame_number += 1

    with open(config, 'r') as f:
        vidat_config = json.load(f)

    output_data = []

    with open(vidat_annotation) as json_file:

        annotations = (json.load(json_file))["annotation"]

        for action_annotation in annotations["actionAnnotationList"]:

            action_label = action_annotation["action"]
            action_label_txt = vidat_config['actionLabelData'][action_label]['name']

            if action_label_txt == 'default':
                action_label_txt = 'null'

            # This is 1-indexed so the 0th second is frame 1
            start_frame = min(
                int(action_annotation["start"] * fps + 1), len(frame_to_timestamp_dict))
            end_frame = min(
                int(action_annotation["end"] * fps + 1), len(frame_to_timestamp_dict))

            start_unix = frame_to_timestamp_dict[start_frame]
            end_unix = frame_to_timestamp_dict[end_frame]

            output_row = (action_label, start_unix, end_unix, action_label_txt)
            output_data.append(output_row)

    with open(out_file, "wt+", newline="") as anno_csv:
        writer = csv.writer(anno_csv, delimiter=",")
        writer.writerows(output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--path', help='File path of directory containing the annotation json and config file')
    parser.add_argument(
        '-f', '--force', help='force overwrite ground truth files', action='store_true')
    parser.add_argument(
        '-c', '--vidat_cfg', default='./configs/vidat_config_munchsonic.json', help='File path of vidat config')

    args = parser.parse_args()

    annotation_files = glob.glob(os.path.join(args.path, '*act*.json'))

    config_path = os.path.join(args.path, 'config.json')
    config = load_config(config_path)

    for i, ref_annotation_path in enumerate(sorted(annotation_files)):
        updated_fname = config['ground_truth']['videos'][i][:-4]
        if (not os.path.exists(os.path.join(args.path, updated_fname + '_records.txt'))) or args.force:
            video = cv2.VideoCapture(os.path.join(
                args.path, updated_fname + '.mp4'))
            fps = video.get(cv2.CAP_PROP_FPS)
            print(f'[INFO] Reference Video FPS: {fps}')

            print('[INFO] Creating class labels using vidat annotation')
            create_labels(args.path, updated_fname, os.path.basename(
                ref_annotation_path), fps, args.vidat_cfg)

            print('[INFO] Creating / updating config file')
            config['ground_truth']['files'] += [
                (updated_fname + '_records.txt')]
    json.dump(config, open(config_path, 'wt'), indent=4)
