#!/bin/bash
set -e

CONFIG_FILE=$1
SAMPLES_FILE=$2
IMAGE_DIR=$3

# parse config file and assign parameters to variables
eval "$(jq -r "to_entries|map(\"export \(.key)=\(.value|tostring)\")|.[]" $CONFIG_FILE)"

# create train job dir
TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S)
TRAIN_JOB_DIR=train_jobs/$TIMESTAMP
mkdir -p $TRAIN_JOB_DIR

# copy config and samples file to train job dir
cp $CONFIG_FILE $TRAIN_JOB_DIR/config.json
cp $SAMPLES_FILE $TRAIN_JOB_DIR/samples.json

# start training
DOCKER_RUN="docker run -d
  -v $IMAGE_DIR:/src/images
  -v "$(pwd)/$TRAIN_JOB_DIR":/src/$TRAIN_JOB_DIR
  -e TRAIN_JOB_DIR=/src/$TRAIN_JOB_DIR
  $docker_image"

eval $DOCKER_RUN

# stream logs from container
CONTAINER_ID=$(docker ps -l -q)
docker logs $CONTAINER_ID --follow
