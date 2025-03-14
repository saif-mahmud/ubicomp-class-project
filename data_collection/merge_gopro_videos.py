import argparse
import json
import os
import subprocess
from datetime import datetime


def get_gopro_video_creation_time(video_path):
    """
    Extracts the creation time of a GoPro video file.

    Args:
        video_path (str): Path to the GoPro video file.

    Returns:
        datetime: Datetime object representing the creation time of the video.
                  Returns None if metadata extraction fails.
    """
    ffprobe_command = ['ffprobe', '-v', 'quiet', '-print_format',
                       'json', '-show_format', '-show_entries', 'format_tags', video_path]
    result = subprocess.run(ffprobe_command, capture_output=True, text=True)
    if result.returncode == 0:
        metadata = json.loads(result.stdout)
        creation_time_str = metadata['format']['tags']['creation_time']
        creation_time = datetime.strptime(
            creation_time_str, '%Y-%m-%dT%H:%M:%S.%f%z')
        return creation_time
    else:
        print(f"Failed to get metadata for {video_path}")
        return None


def write_metadata_to_video(input_video_path, output_video_path, creation_time):
    """
    Writes metadata to a new video file.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to the output video file.
        creation_time (datetime): Datetime object representing the creation time of the video.
    """
    ffmpeg_command = [
        'ffmpeg', '-i', input_video_path,
        '-metadata', f'creation_time={creation_time.strftime("%Y-%m-%dT%H:%M:%S%z")}',
        '-codec', 'copy', output_video_path
    ]
    subprocess.run(ffmpeg_command)


def concatenate_videos(input_dir, output_dir):
    """
    Concatenates GoPro video files in a directory.

    Args:
        input_dir (str): Path to the input directory containing GoPro video files.
        output_dir (str): Path to the output directory to save the concatenated video file.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # Get list of .MP4 files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.MP4')]
    if not video_files:
        print("No .MP4 files found in the input directory.")
        return

    # Sort the video files by creation time
    video_files_sorted = sorted(video_files)

    # Get creation datetime of the first video
    first_video_path = os.path.join(input_dir, video_files_sorted[0])
    creation_datetime = get_gopro_video_creation_time(first_video_path)

    # Generate input file list for ffmpeg concat demuxer
    input_file_list_path = os.path.join(output_dir, 'input_list.txt')
    with open(input_file_list_path, 'w') as f:
        for file_name in video_files_sorted:
            f.write(f"file '{os.path.join(input_dir, file_name)}'\n")

    # Build ffmpeg command
    _out_path = os.path.join(output_dir, f"tmp_{creation_datetime}.mp4")
    out_path = os.path.join(
        output_dir, f"{creation_datetime.strftime('%Y%m%d_%H%M%S')}.mp4")
    print(f'\n[INFO] {out_path} <--- {video_files_sorted}')

    # Build ffmpeg command
    ffmpeg_command = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', input_file_list_path,
        '-c', 'copy', '-map_metadata', '0', _out_path
    ]

    # Print the ffmpeg command
    print("\033[92m\nExecuting ffmpeg command:\033[0m")
    print(" ".join(ffmpeg_command))

    # Run ffmpeg command to concatenate videos
    subprocess.run(ffmpeg_command)

    # Add original metadata to merged video (required for future processing)
    print(f'\033[92m\n\n[INFO] Writing metadata to the new video\033[0m')
    if creation_datetime:
        write_metadata_to_video(_out_path, out_path, creation_datetime)

    # Remove the input file list
    os.remove(input_file_list_path)
    os.remove(_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_directory',
                        help='File path of directory containing the .MP4 files recorded by GoPro')
    parser.add_argument(
        '-o', '--output_directory', help='File path of the output directory to save the merged .mp4 file')

    args = parser.parse_args()

    concatenate_videos(args.input_directory, args.output_directory)
