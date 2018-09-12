#!/usr/bin/env bash
##########################################################
# where to write tfevents
OUTPUT_DIR="gs://model-exports"
# experiment settings
TRAIN_BATCH=512
EVAL_BATCH=512
LR=0.001
EPOCHS=100
# create a job name for the this run
prefix="example"
now=$(date +"%Y%m%d_%H_%M_%S")
JOB_NAME="$ENV_NAME"-"$prefix"_"$now"
# locations locally or on the cloud for your files
TRAIN_FILES="data/train.tfrecords"
EVAL_FILES="data/val.tfrecords"
TEST_FILES="data/test.tfrecords"
##########################################################

if [[ -z $0 && -z $1 ]]; then
    echo "Incorrect arguments specified."
    echo ""
    echo "Usage: ./train_local_cpu.sh [ENV_NAME]"
    echo ""
    exit 1
else
    if [[ -z $0 ]]; then
        ENV_NAME="default"
    else
        ENV_NAME=$1
    fi
fi

# needed to use virtualenvs
set -euo pipefail

# get current working directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# create folders if they don't exist of logs and outputs
mkdir -p ${DIR}/runlogs

# create a local job directory for checkpoints etc
JOB_DIR=${OUTPUT_DIR}/${JOB_NAME}

###################
# Add notes to the log file based on the current information about this training job close vim to start training
# useful if you are running lots of different experiments and you forget what values you used
echo "---
## ${JOB_NAME}" >> training_log.md
echo "Learning Rate: ${LR}" >> training_log.md
echo "Epochs: ${EPOCHS}" >> training_log.md
echo "Batch Size (train/eval): ${TRAIN_BATCH}/ ${EVAL_BATCH}" >> training_log.md
echo "### Hypothesis
" >> training_log.md
echo "### Results
" >> training_log.md
vim + training_log.md
###################

# activate the virtual environment
if [[ -z $1 ]]; then
    set +u
    source ${ENV_NAME}/bin/activate
    set -u
fi

export CUDA_VISIBLE_DEVICES=""
# start training
python3 -m initialisers.task \
        --job-dir ${JOB_DIR} \
        --train-batch-size ${TRAIN_BATCH} \
        --eval-batch-size ${EVAL_BATCH} \
        --learning-rate ${LR} \
        --num-epochs ${EPOCHS} \
        --train-files ${TRAIN_FILES} \
        --eval-files ${EVAL_FILES} \
        --test-files ${TEST_FILES} \
        --export-path "${OUTPUT_DIR}exports" \
               &>runlogs/$2.log &
               echo "$!" > runlogs/$2.pid
