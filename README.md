#  Detection of Situational Impairments Using Acoustic Sensing
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


## Data Collection w/GoPro

1.  Create a directory ```<Pxx>-wild``` in ```data/pilot_study``` and copy the ```.MP4``` video file(s) from GoPro SD card to there.

2. Change working directory to ```data_collection```

```shell
cd data_collection
```

Then run the following Python script:

```shell
python gopro_data_postprocessing.py --count_down 15 --folds 5 --path ../data/pilot_study/<Pxx>
```

3. Open the newly created "record_*.mp4" video (can be found in ```data/pilot_study/<Pxx>```) in [Vidat](https://vidat2.davidz.cn/#/).

4. Click the triple bar icon in the top left and navigate to **Configuration**. Upload the ```data_collection/configs/vidat_config_activity_all.json``` file and navigate back to **Annotation**.

5. Label the video segments accordingly (instructions for how to do so are [here](https://github.com/anucvml/vidat)). Save the annotation file as ```act_<Pxx>_<vid_seq_num>.json``` (where ```<vid_seq_num>``` is the video sequence number sorted by time) to the ```<Pxx>``` is the participant id. For example, if you are annotating the second video (sorted in ascending order of time) of the participant ```P02```, the name of the annotation file will be ```act_P02_02.json```.

6. Run the following Python script to create ground truth files compatible with machine learning system:

```shell
python create_vidat_label.py --force --path ../data/pilot_study/<Pxx>
```

### Merging audio file(s) collected through BLE SD Module (nRF52840)

First copy all the .raw audio files from SD card (of format ```audioXXX.000.raw```, ```audioXXX.001.raw```, ..., ```audioYYY.000.raw```, ..., ```audioYYY.00N.raw```) to ```tmp-data```. Then, run the following command to merge the componenet audio files into one .raw file: 

```shell
python merge_audio_files.py --raw_files_dir ../data/tmp-data --out_dir ../data/pilot_study/<Pxx>
```

The merged audio file(s) will be dumped to the ```out_dir``` directory.


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

### Pushing Collected Data to Server

```shell
rsync -rav --exclude dataset/ <Pxx> lablan:/data/saif/acoustrap/pilot_study
```

## Machine Learning Model

* Change working directory to ```ml_model```

### ResNet Model Training

```shell
python training.py --model resnet18 --epochs 50 --batch_size 64 --sliding_window_size 2.0 --gpus 0
```
