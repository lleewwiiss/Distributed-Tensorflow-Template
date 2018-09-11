#!/bin/bash
if [ $# -ne 2 ]; then
    echo $0: usage: train_local_single envname gpu_id
    exit 1
fi

# ensure you are linked to cuda on the machine
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda

# needed to use virtualenvs
set -euo pipefail

# create a job name for the this run
prefix="example"
now=$(date +"%Y%m%d_%H:%M:%S")
JOB_NAME=${1-"${prefix}_${now}"}

# link to a bucket on gsp
GCS_BUCKET="gs://example-bucket/"

# Batch size
BATCH=512
# learning rate
LR="0.001"
# number of epochs
EPOCHS="100"

# locations locally or on the cloud for your files
TRAIN_FILES="data/train.tfrecords"
EVAL_FILES="data/val.tfrecords"
TEST_FILES="data/test.tfrecords"

# get current working directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# create folders if they don't exist of logs and outputs
mkdir -p $DIR/jobs
mkdir -p $DIR/runlogs

# create a local job directory for checkpoints etc
JOB_DIR=$DIR/jobs/$JOB_NAME

# Add notes to the log file based on the current information about this training job close vim to start training
# useful if you are running lots of different experiments and you forget what values you used
echo "---  
## $JOB_NAME" >> training_log.md
echo "Learning Rate: $LR" >> training_log.md
echo "Epochs: $EPOCHS" >> training_log.md
echo "Batch Size (train/eval): $BATCH / $BATCH" >> training_log.md
echo "### Hypothesis
" >> training_log.md
echo "### Results
" >> training_log.md
vim + training_log.md

# activate the virtual environment
set +u
source $1/bin/activate
set -u

# start training
export CUDA_VISIBLE_DEVICES="$1" #SET TO GPU number
python3 -m initialisers.task \
        --job-dir ${JOB_DIR} \
        --train-batch-size ${BATCH} \
        --eval-batch-size ${BATCH} \
        --learning-rate ${LR} \
        --num-epochs ${EPOCHS} \
        --train-files ${TRAIN_FILES} \
        --eval-files ${EVAL_FILES} \
        --test-files ${TEST_FILES} \
        --export-path "${GCS_BUCKET}exports" \
               &>runlogs/$2.log &
               echo "$!" > runlogs/$2.pid
