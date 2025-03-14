'''
Visualize echo profile patterns for a certain gesture from a data collection
7/11/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import cv2
import json
import argparse
import numpy as np

from load_save_gt import load_gt
from utils import load_frame_time
from plot_profiles import plot_profiles_split_channels

def extract_gestures(dataset_parent, gois, echo_profile_suffix='fmcw_16bit_diff', maxval=None, minval=None):

    all_goi_echo_profiles = None

    label_to_goi_index = {}
    
    goi_texts = {}

    config = json.load(open(os.path.join(dataset_parent, 'config.json'), 'rt'))
    audio_config = config['audio']['config']

    n_channels = len(audio_config['channels_of_interest'])
    if isinstance(audio_config['tx_file'], list):
        n_channels *= len(audio_config['tx_file'])

    for n in range(len(config['ground_truth']['files'])):
        gt_file = os.path.join(dataset_parent, config['ground_truth']['files'][n])
        echo_profile_file = os.path.join(dataset_parent, '%s_%s_profiles.npy' % (config['audio']['files'][n][:-4], echo_profile_suffix))

        gt_syncing_pos = config['ground_truth']['syncing_poses'][n]
        if gt_syncing_pos < 1600000000:
            gt_ts_file = os.path.join(dataset_parent, config['ground_truth']['videos'][n][:-4] + '_frame_time.txt')
            video_ts = load_frame_time(gt_ts_file)
            gt_syncing_pos = video_ts[gt_syncing_pos - 1]
        audio_syncing_frame = config['audio']['syncing_poses'][n]
        if audio_syncing_frame >= audio_config['frame_length'] * 50:
            audio_syncing_frame = audio_syncing_frame // audio_config['frame_length']

        def ts_to_profile_idx(ts):
            return round((ts - gt_syncing_pos) * audio_config['sampling_rate'] / audio_config['frame_length'] + audio_syncing_frame)

        recording_gt = load_gt(gt_file)
        recording_profile = np.load(echo_profile_file)

        if all_goi_echo_profiles is None:
            if -1 in gois:
                gois = list(range(max([int(x[0]) for x in recording_gt]) + 1))
            all_goi_echo_profiles = [[] for _ in gois]
            for i, g in enumerate(gois):
                label_to_goi_index[g] = i
        
        for i in range(len(all_goi_echo_profiles)):
            all_goi_echo_profiles[i] += [[]]

        for truth in recording_gt:
            if int(truth[0]) in label_to_goi_index:
                # print(truth[1], truth[2], ts_to_profile_idx(float(truth[1])), ts_to_profile_idx(float(truth[2])))
                # print(n, echo_profile_file, recording_profile.shape)
                if ts_to_profile_idx(float(truth[1])) >= recording_profile.shape[1] or ts_to_profile_idx(float(truth[2])) >= recording_profile.shape[1]:
                    continue
                gesture_profile = recording_profile[:, ts_to_profile_idx(float(truth[1])): ts_to_profile_idx(float(truth[2]))]
                gesture_profile_pattern = plot_profiles_split_channels(gesture_profile, n_channels, maxval, minval)
                all_goi_echo_profiles[label_to_goi_index[int(truth[0])]][-1] += [gesture_profile_pattern]
                goi_texts[int(truth[0])] = truth[3]

    return all_goi_echo_profiles, goi_texts


def visualize_gesture_patterns(dataset_parent, gois, width, echo_profile_suffix='fmcw_16bit_diff', output_suffix='', maxval=None, minval=None):

    # shape: n_goi x n_recordings x n_reps_per_rcd x (h x w, w varies)
    all_goi_echo_profiles, goi_texts = extract_gestures(dataset_parent, gois, echo_profile_suffix, maxval, minval)

    if -1 in gois:
        gois = list(range(len(goi_texts)))
    max_width = max([max([max([rep.shape[1] for rep in rcd]) for rcd in gesture]) for gesture in all_goi_echo_profiles])
    max_gestures_per_line = max([max([len(rcd) for rcd in gesture]) for gesture in all_goi_echo_profiles])
    gesture_width = max(max_width, width) + 2
    overall_width = max_gestures_per_line * gesture_width
    gesture_height = all_goi_echo_profiles[0][0][0].shape[0] + 2
    overall_height = gesture_height * len(all_goi_echo_profiles[0])

    os.makedirs(os.path.join(dataset_parent, 'gesture_vis'), exist_ok=True)

    for i, gesture_profiles in enumerate(all_goi_echo_profiles):
        gesture_profile_image = np.zeros((overall_height, overall_width, 3), dtype=np.uint8)
        for y, recording_profile in enumerate(gesture_profiles):
            for x, rep_profile in enumerate(recording_profile):
                v_start = y * gesture_height + 1
                h_start = x * gesture_width + (gesture_width - rep_profile.shape[1]) // 2
                gesture_profile_image[v_start: v_start + rep_profile.shape[0], h_start: h_start + rep_profile.shape[1], :] = rep_profile

        cv2.imwrite(os.path.join(dataset_parent, 'gesture_vis', 'gesture_%d_%s_%s_%s.png' % (gois[i], goi_texts[gois[i]], echo_profile_suffix, output_suffix)), gesture_profile_image)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Echo profile visualization (split gestures)')
    parser.add_argument('-p', '--path', help='path to the audio file, .wav or .raw')
    parser.add_argument('--goi', help='gestures of interest, comma-separated, -1 for all', default='-1')
    parser.add_argument('-w', '--width', help='width of each gesture', type=int, default=0)
    parser.add_argument('--profile', help='profile suffix', default='fmcw_16bit_diff')
    parser.add_argument('-o', '--output', help='output suffix', default='')
    parser.add_argument('-md', '--maxval', help='maxval for profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-nd', '--minval', help='maxval for profiles figure rendering, 0 for adaptive', type=float, default=0)

    args = parser.parse_args()

    visualize_gesture_patterns(args.path, [int(x) for x in args.goi.split(',')], args.width, args.profile, args.output, args.maxval, args.minval)