import argparse
import json
import os
import re
import warnings

import numpy as np

from echo_profiles import echo_profiles
from load_save_gt import load_gt, save_gt
# from mediapipe_pose import mediapipe_pose_estimation
from body_landmarks_hmr import body_landmarks_hmr
from utils import load_frame_time


def load_config(parent_folder, config_file):
    if len(config_file) == 0:
        config_file = os.path.join(parent_folder, 'config.json')
    if not os.path.exists(config_file):
        raise FileNotFoundError('Config file at %s not found' % config_file)
    config = json.load(open(config_file, 'rt'))
    assert (len(config['audio']['files']) == len(
        config['audio']['syncing_poses']))  # every audio file should have a sync mark
    assert (len(config['ground_truth']['files']) == len(
        config['sessions']))  # every ground truth file should have its session config
    assert (len(config['audio']['files']) == len(
        config['ground_truth']['files']))  # at this moment, audio files and gt files should have same amount

    if len(config['audio']['config']['channels_of_interest']) == 0:
        config['audio']['config']['channels_of_interest'] = list(
            range(config['audio']['config']['n_channels']))

    for f in config['ground_truth']['files']:
        if not os.path.exists(os.path.join(parent_folder, f)):
            # ground truth file must exist
            raise FileNotFoundError('Ground truth file at %s not found' % f)

    if 'videos' not in config['ground_truth'] or len(config['ground_truth']['videos']) == 0:
        gt_videos = []
        for f in config['ground_truth']['files']:
            matched_video = re.findall(r'(\w+\d{6})', f)
            if not matched_video:
                warnings.warn('Warning: could not automatically detect video file for ground truth file %s' % f,
                              RuntimeWarning)
                continue
            gt_videos += [matched_video[0] + '.mp4']
        config['ground_truth']['videos'] = gt_videos
    else:
        assert (len(config['ground_truth']['files']) == len(config['ground_truth'][
            'videos']))  # if you want to specify videos manually, make sure that you cover all of them

    if 'tasks' not in config or len(config['tasks']) == 0:
        config['tasks'] = ['classification', 'pose_estimation']  # default task is classification
    return config


def ts_to_idx(ts, all_ts):
    return np.argmin(np.abs(all_ts - ts))


def data_preparation(parent_folder, config_file='', force_overwrite=False, response_offset=0.2, target_bitwidth=16,
                     maxval=None, maxdiff=None, mindiff=None, rectify=False, no_overlapp=False, no_diff=False):
    config = load_config(parent_folder, config_file)
    # deal with audios first
    audio_config = config['audio']['config']
    for f in config['audio']['files']:
        audio_path = os.path.join(parent_folder, f)
        if not os.path.exists(audio_path):
            warnings.warn('Warning: audio file %s is specified in config file but was not found, skipped' % f,
                          RuntimeWarning)
            continue
        if (not os.path.exists('%s_%s_%dbit_profiles.npy' % (
                audio_path[:-4], audio_config['signal'].lower(), target_bitwidth))) or force_overwrite:
            echo_profiles(audio_path, audio_config, target_bitwidth, maxval, maxdiff, mindiff, rectify, no_overlapp,
                          no_diff)

    # generate pose landmarks based on video
    if 'pose_estimation' in config['tasks']:
        for f in config['ground_truth']['videos']:
            video_path = os.path.join(parent_folder, f)
            if not os.path.exists(video_path):
                warnings.warn('Warning: video file %s is specified in config file but was not found, skipped' % f,
                              RuntimeWarning)
                continue

            gt_pose_file_path = f'{video_path[:-4]}_pose_landmarks.npy'
            gt_pose_video_path = f'{video_path[:-4]}_pose_video.mp4'
            video_frame_timestamp_unix = f'{video_path[:-4]}_frame_time.txt'

            if not os.path.exists(gt_pose_file_path):
                # print(f'Generating body pose landmarks (using Mediapipe) for {video_path}') 
                # mediapipe_pose_estimation(source_video_path=video_path,
                #                           target_video_path=gt_pose_video_path,
                #                           timestamp_file_path=video_frame_timestamp_unix,
                #                           ground_truth_file_path=gt_pose_file_path)
                
                print(f'Generating SMPL body pose (using HMR2) for {video_path}')
                body_landmarks_hmr(source_video_path=video_path, 
                                   timestamp_file_path=video_frame_timestamp_unix, 
                                   target_video_path=gt_pose_video_path, 
                                   ground_truth_file_path=gt_pose_file_path, 
                                   hmr_gpu=0, detectron_gpu=0, batch_size=16)

    # find all the _profiles.npy files
    all_profile_npys = [x for x in os.listdir(
        parent_folder) if x[-13:] == '_profiles.npy']

    for n_rcd, rcd_sessions in enumerate(config['sessions']):
        # syncing poses in the format of frames
        gt_syncing_pos = config['ground_truth']['syncing_poses'][n_rcd]
        if abs(gt_syncing_pos - round(gt_syncing_pos)) < 1e-6:
            gt_ts_file = os.path.join(
                parent_folder, config['ground_truth']['videos'][n_rcd][:-4] + '_frame_time.txt')
            gt_video_ts = load_frame_time(gt_ts_file)
            gt_syncing_pos = gt_video_ts[gt_syncing_pos - 1]
            video_start_ts = gt_video_ts[0]
        else:
            video_start_ts = gt_syncing_pos
        audio_syncing_pos = config['audio']['syncing_poses'][n_rcd]
        if audio_syncing_pos < config['audio']['config']['frame_length'] * 50:
            audio_syncing_pos *= config['audio']['config']['frame_length']

        rcd_config = {
            'audio_config': audio_config,
            'syncing': {
                'audio': audio_syncing_pos,
                'ground_truth': gt_syncing_pos,
            },
            'response_offset': response_offset
        }

        gts = []
        if 'classification' in config['tasks']:
            gt_file = os.path.join(
                parent_folder, config['ground_truth']['files'][n_rcd])
            gt_classification = load_gt(gt_file)
            gt_start_ts = np.array([float(x[1]) for x in gt_classification])
            gt_end_ts = np.array([float(x[2]) for x in gt_classification])
            gts += [{
                'gt': gt_classification,
                'target': 'ground_truth_classification.txt',
                'start_ts': gt_start_ts,
                'end_ts': gt_end_ts
            }]

        if 'pose_estimation' in config['tasks']:
            gt_file = os.path.join(parent_folder, config['ground_truth']['videos'][n_rcd])[
                :-4] + '_pose_landmarks.npy'
            gt_landmarks = load_gt(gt_file)
            gt_ts = np.array([x[0] for x in gt_landmarks])
            gts += [{
                'gt': gt_landmarks,
                'target': 'ground_truth_pose_landmarks.npy',
                'start_ts': gt_ts,
                'end_ts': gt_ts
            }]

        for n_ss, ss in enumerate(rcd_sessions):
            print('Dealing with session %02d in recording %02d, ' %
                  (n_ss + 1, n_rcd + 1), end='')
            session_target = os.path.join(
                parent_folder, 'dataset', 'session_%02d%02d' % (n_rcd + 1, n_ss + 1))

            if not os.path.exists(session_target):
                os.makedirs(session_target)

            if ss['start'] < 1600000000:
                ss_s = video_start_ts + ss['start']
            else:
                ss_s = ss['start']

            ss_e = ss_s + ss['duration']

            # writing and linking files
            config_target = os.path.join(session_target, 'config.json')
            print('    Writing session config at %s' % config_target)
            json.dump(rcd_config, open(config_target, 'wt'), indent=4)

            for npy in all_profile_npys:
                if npy[:len(config['audio']['files'][n_rcd]) - 4] == config['audio']['files'][n_rcd][:-4]:
                    target_npy = os.path.join(
                        session_target, npy[len(config['audio']['files'][n_rcd]) - 3:])
                    if not os.path.exists(target_npy):
                        print('    Linking %s -> %s' % (npy, target_npy))
                        os.symlink(os.path.abspath(
                            os.path.join(parent_folder, npy)), target_npy)

            for this_gt in gts:
                ss_s_idx = ts_to_idx(ss_s, this_gt['start_ts'])
                ss_e_idx = ts_to_idx(ss_e, this_gt['end_ts'])
                session_gt = this_gt['gt'][ss_s_idx: ss_e_idx + 1]
                # print('ground truth length: %d' % (len(session_gt)))

                gt_target = os.path.join(session_target, this_gt['target'])
                print('    Writing ground truth at %s, length %d' %
                      (gt_target, len(session_gt)))
                save_gt(session_gt, gt_target)

            ref_video_source = os.path.join(
                parent_folder, config['ground_truth']['videos'][n_rcd])
            ref_video_target = os.path.join(
                session_target, 'reference_video.mp4')
            if os.path.exists(ref_video_source) and (not os.path.exists(ref_video_target)):
                print('    Linking %s -> %s' %
                      (config['ground_truth']['videos'][n_rcd], ref_video_target))
                os.symlink(os.path.abspath(ref_video_source), ref_video_target)
                os.symlink(os.path.abspath(ref_video_source[:-4] + '_frame_time.txt'),
                           ref_video_target[:-4] + '_frame_time.txt')

            pose_landmarks_source = os.path.join(parent_folder, config['ground_truth']['videos'][n_rcd][
                :-4] + '_pose_landmarks.npy')
            pose_landmarks_target = os.path.join(
                session_target, 'ground_truth_pose_landmarks.npy')
            if os.path.exists(pose_landmarks_source) and (not os.path.exists(pose_landmarks_target)):
                print('    Linking %s -> %s' % (
                    config['ground_truth']['videos'][n_rcd][:-4] + '_pose_landmarks.npy', pose_landmarks_target))
                os.symlink(os.path.abspath(pose_landmarks_source),
                           pose_landmarks_target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Pre-process data and generate dataset for training')
    parser.add_argument(
        '-p', '--path', help='path to the folder where files are saved (and dataset will be saved)')
    parser.add_argument(
        '-c', '--config', help='path to the config.json file', default='')
    parser.add_argument(
        '-f', '--force', help='force overwrite echo profile files', action='store_true')
    parser.add_argument(
        '--response_offset', help='response time offset (s)', type=float, default=0.2)
    parser.add_argument('-tb', '--target_bitwidth',
                        help='target bitwidth, 2-16', type=int, default=16)
    parser.add_argument('-m', '--maxval', help='maxval for original profiles figure rendering, 0 for adaptive',
                        type=int, default=0)
    parser.add_argument('-md', '--maxdiffval', help='maxval for differential profiles figure rendering, 0 for adaptive',
                        type=int, default=0)
    parser.add_argument('-nd', '--mindiffval', help='maxval for differential profiles figure rendering, 0 for adaptive',
                        type=int, default=0)
    parser.add_argument('-r', '--rectify',
                        help='rectify speaker curve', action='store_true')
    parser.add_argument(
        '--no_overlapp', help='no overlapping while processing frames', action='store_true')
    parser.add_argument(
        '--no_diff', help='do not generate differential echo profiles', action='store_true')
    # parser.add_argument('--stack', help='stack multiple frames or split frames', action='store_true')
    # parser.add_argument('--fr', help='frame_ratio', type=int, default=1)

    args = parser.parse_args()
    data_preparation(args.path, args.config, args.force, args.response_offset, args.target_bitwidth, args.maxval,
                     args.maxdiffval, args.mindiffval, args.rectify, args.no_overlapp, args.no_diff)
