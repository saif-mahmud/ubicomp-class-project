import math
import random

import cv2
import numpy as np


def generate_discrete(cmds, folds, reps_per_fold, shuffle: bool = True):
    original_cmd_list = []
    img_list = {}
    uses_vid = False

    for i, (cmd_idx, cmd) in enumerate(zip(cmds['idx'], cmds['cmds'])):
        if len(cmds['instruction_imgs']) == 0:
            original_cmd_list += [(cmd_idx, cmd, None, cmds['durations'][i])]
        # Uses videos instead of images
        elif any('.mp4' in img for img in cmds['instruction_imgs']):
            uses_vid = True
            original_cmd_list += [(cmd_idx, cmd, cmds['instruction_imgs'][i])]
            img_list[cmds['instruction_imgs'][i]] = cv2.VideoCapture(
                'img/%s' % cmds['instruction_imgs'][i])
        else:
            original_cmd_list += [(cmd_idx, cmd, cmds['instruction_imgs'][i])]
            img_list[cmds['instruction_imgs'][i]] = cv2.resize(cv2.imread('img/%s' % cmds['instruction_imgs'][i]),
                                                               (331, 400))

    cmd_list = []
    cmds_repped = original_cmd_list * reps_per_fold
    for _ in range(folds):
        if shuffle:
            random.shuffle(cmds_repped)
        cmd_list += cmds_repped

    return cmd_list, img_list, uses_vid
