#!/bin/sh

# API Key:
# HCMKg0jqdE3HWVrI2XHvc488vcuHpYAb7JxpoI7tuwGT

# URL:
# https://api.us-east.text-to-speech.watson.cloud.ibm.com/instances/f0335466-710e-4420-b8b4-069c9d7ecd55

in_fname='dribble_left.wav'
out_fname='./audio/dribble_left.mp3'


curl -X POST -u "apikey:HCMKg0jqdE3HWVrI2XHvc488vcuHpYAb7JxpoI7tuwGT" \
--header "Content-Type: application/json" \
--header "Accept: audio/wav" \
--data "{\"text\":\"Dribble Left\"}" \
--output $in_fname \
"https://api.us-east.text-to-speech.watson.cloud.ibm.com/instances/f0335466-710e-4420-b8b4-069c9d7ecd55/v1/synthesize?voice=en-US_MichaelV3Voice"


ffmpeg -i $in_fname -vn -ar 44100 -ac 2 -b:a 192k $out_fname
rm -rf $in_fname