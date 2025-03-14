#  UbiComp Class Project - Team Sonic
## Development Environment Setup

Create a Python environment using conda:

```shell
conda create -n team-sonic python=3.9
```

Then install the dependencies using this command:

```shell
pip install -r requirements.txt
```

## Data Collection w/Computer

* Change working directory to ```data_collection```

```shell
cd data_collection
```

_**N.B.** All the directory changes in this doc is on the assumption that you are currently at project root._

```shell
python data_collection.py --path-prefix ../data/pilot_study/ --commandsets upper-body --count_down 15 --folds 10 --reps_per_fold 1 --noserial --output <Pxx>
```

Here, ```<Pxx>``` is the participant id (e.g. P01 / P02 / P03 etc.)

## Data Processing

* Change working directory to ```data_preparation```

```shell
cd data_preparation
```

### Syncing acoustic signal with video data

1. Copy the ```.raw``` audio file(s) from SD card to the directory specified through ```--output``` flag in the data
   collection script.
2. Find out the **frame number** in recorded video file where the participant clapped. Write / add the frame number in
   the ```"syncing_poses"``` list under the ```"ground_truth"``` field of ```config.json``` file.

```shell
python audio_auto_sync.py --path ../data/pilot_study/<Pxx>
```

### Data Preparation for training

**if** the data was collected using **Teensy 4.1**:
```shell
python data_preparation.py -md 500000000 -nd -500000000 -f --path ../data/pilot_study/<Pxx>
```

**if** the data was collected using **BLE SD Module (nRF52840)**:
```shell
python data_preparation.py -md 120000000 -nd -120000000 -f --path ../data/pilot_study/<Pxx>
```

### Echo Profile Visualization

```shell
python visualize_echo_profiles.py --height 605 --path ../data/pilot_study/<Pxx>
```
