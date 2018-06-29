#!/bin/bash
set -e

# start training
python -W ignore -m trainer.train -j /src/$TRAIN_JOB_DIR -i /src/images
