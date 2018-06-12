#!/bin/bash
set -e

# start training
python -W ignore -m trainer.train -j /src/$TRAIN_JOB_DIR -i /src/images

# copy train output to s3
aws s3 cp /src/$TRAIN_JOB_DIR s3://$S3_BUCKET/$TRAIN_JOB_DIR --recursive
