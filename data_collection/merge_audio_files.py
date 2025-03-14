import argparse
import os
import subprocess
from collections import defaultdict


def group_audio_files(file_list):
    """
    Groups a list of audio file names based on their base names.

    Parameters:
        file_list (list): A list of strings representing audio file names.

    Returns:
        dict: A dictionary where keys are base names of the audio files
              and values are lists of corresponding file names.
    """
    groups = defaultdict(list)
    for filename in file_list:
        base_name = filename.split('.')[0]
        groups[base_name].append(filename)

    return dict(groups)


def remove_empty_files(data_dir, file_list):
    """
    Removes empty files from a list of file paths.

    Parameters:
        data_dir (str): The directory where the files are located.
        file_list (list): A list of strings representing file paths.

    Returns:
        list: A list of non-empty file paths.
    """
    return [os.path.join(data_dir, fpath) for fpath in file_list if os.path.getsize(os.path.join(data_dir, fpath)) > 0]


def concat_audio_files(file_list, output_file):
    """
    Concatenates audio files into a single file.

    Parameters:
        file_list (list): A list of strings representing file paths.
        output_file (str): The path to the output file.

    Returns:
        None
    """
    subprocess.run(['cat'] + file_list, stdout=open(output_file, 'wb'))


def merge_audio_files(raw_files_dir, out_dir, base_name='audio'):
    """
    Merges audio files into a single file per base name.

    Parameters:
        raw_files_dir (str): The directory containing the raw audio files.
        out_dir (str): The directory where the merged files will be stored.
        base_name (str, optional): The base name of the audio files. Defaults to 'audio'.

    Returns:
        None
    """
    raw_files = sorted([file for file in os.listdir(
        raw_files_dir) if file.startswith(base_name) and file.endswith('.raw')])
    grouped_file_dict = group_audio_files(raw_files)

    for base_audio in grouped_file_dict.keys():
        out_fpath = os.path.join(out_dir, base_audio + '.raw')
        seg_fpaths = remove_empty_files(
            raw_files_dir, grouped_file_dict[base_audio])

        os.makedirs(out_dir, exist_ok=True)
        if seg_fpaths:
            print(out_fpath, '->', seg_fpaths)
            concat_audio_files(file_list=seg_fpaths, output_file=out_fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--raw_files_dir',
                        help='File path of directory containing the .raw files')
    parser.add_argument(
        '-o', '--out_dir', help='File path of the output directory to save the merged .raw file')

    args = parser.parse_args()

    merge_audio_files(raw_files_dir=args.raw_files_dir, out_dir=args.out_dir)
