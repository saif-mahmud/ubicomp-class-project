#!/bin/bash
python data_collection.py --path-prefix ../data/user_study/ --commandsets full-body --count_down 15 --folds 5 --reps_per_fold 1 --noserial --play_audio --output $1