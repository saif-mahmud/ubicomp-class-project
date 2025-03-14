#!/bin/bash
python data_collection.py --path-prefix ../data/user_study/ --commandsets hand-face-gesture --count_down 30 --folds 3 --reps_per_fold 1 --noserial --play_audio --output $1