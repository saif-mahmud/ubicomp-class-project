#!/bin/bash
python data_collection.py --path-prefix ../data/pilot_study/ --commandsets upper-body --count_down 10 --folds 1 --reps_per_fold 1 --noserial --play_audio --output $1