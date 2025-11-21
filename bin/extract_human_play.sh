#!/bin/bash

# Edit here
# human_play_dataset=/path/to/human-play/lerobot/dataset
human_play_dataset=/home/rnkj/dataset/HRC/handpose/_test_human
camera_view1=front
camera_view2=side

# If you have installed conda in a different path, edit here
PATH_TO_CONDA_ACTIVATE=$HOME/miniforge3/bin/activate
HAND_OBJ_CONDA_ENV=handobj
LEROBOT_CONDA_ENV=lerobot

# Extract human play via hand_object_detector
source $PATH_TO_CONDA_ACTIVATE $HAND_OBJ_CONDA_ENV

# convert av1 mp4 to h264 mp4 via ffmpeg
mkdir /tmp/videos
for mp4 in $(find $human_play_dataset/videos -name "*.mp4"); do
    output_mp4=$(echo $mp4 | sed "s|$human_play_dataset/videos|/tmp/videos|")
    mkdir -p $(dirname $output_mp4)
    ffmpeg -loglevel quiet -i $mp4 -c:v libx264 $output_mp4
    ffmpeg -i $mp4 -c:v libx264 $output_mp4
done

conda deactivate

# Create LeRobot dataset of human play
source $PATH_TO_CONDA_ACTIVATE $LEROBOT_CONDA_ENV
conda deactivate
