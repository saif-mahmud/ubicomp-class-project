#!/bin/bash

# Check if a directory argument was provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <video-directory>"
  exit 1
fi

video_dir="$1"

# Verify the provided argument is a directory
if [ ! -d "$video_dir" ]; then
  echo "Error: '$video_dir' is not a directory."
  exit 1
fi

# Loop over all .mp4 files in the directory
for file in "$video_dir"/*.mp4; do
  # Skip if no file is found
  [ -e "$file" ] || continue

  echo "Compressing: $file"
  # Create a temporary file name based on the original
  tmpfile="${file%.*}_temp.${file##*.}"

  # Compress the video with ffmpeg using CRF mode and slow preset
  ffmpeg -i "$file" -c:v libx264 -preset slow -crf 28 -c:a copy "$tmpfile"

  # If the compression is successful, replace the original file with the compressed one
  if [ $? -eq 0 ]; then
    mv -f "$tmpfile" "$file"
    echo "Replaced original: $file"
  else
    echo "Error processing $file, original file preserved."
    rm -f "$tmpfile"
  fi
done
