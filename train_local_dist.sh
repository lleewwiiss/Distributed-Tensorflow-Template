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
    echo "Usage: ./train_local_dist.sh [ENV_NAME]"
    echo ""
    exit 1
else
    if [[ -z $0 ]]; then
        ENV_NAME="default"
    else
        ENV_NAME=$1
    fi
fi

if [[ -z $LD_LIBRARY_PATH || -z $CUDA_HOME  ]]; then
    echo ""
    echo "CUDA environment variables not set."
    echo "Consider adding them to your shell-rc."
    echo ""
    echo "Example:"
    echo "----------------------------------------------------------"
    echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"'
    echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"'
    echo 'CUDA_HOME="/usr/local/cuda"'
    echo ""
fi

# needed to use virtualenvs
set -euo pipefail

# get current working directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# create folders if they don't exist of logs and outputs
mkdir -p $DIR/runlogs

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

# Setup the distributed workflow. Ideally you would like at least twice as many workers as parameter servers, and
# each worker have a gpu associate with it, ps = parameter server

# This is an example for 3 GPUS, mocking the cloud training environment. The two workers use 2 GPUs and the master 1.
# Make sure specified ports are not being used
config="
{
    \"master\": [\"localhost:27182\"],
    \"ps\": [\"localhost:27183\"],
    \"worker\": [
        \"localhost:27184\",
        \"localhost:27185\"
        ]
}, \"environment\": \"cloud\""

echo "Starting Training"

function run {
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
               &>runlogs/$1.log &
               echo "$!" > runlogs/$1.pid

}

# activate the virtual environment
if [[ -z $1 ]]; then
    set +u
    source $ENV_NAME/bin/activate
    set -u
fi

# ensure parameter server doesn't use any of the GPUS
export CUDA_VISIBLE_DEVICES=""
# Parameter Server can be run on cpu
task="{\"type\": \"ps\", \"index\": 0}"
export TF_CONFIG="{\"cluster\":${config}, \"task\":${task}}"
run ps

# Master can be run on GPU as it runs the evaluation
export CUDA_VISIBLE_DEVICES="1"
task="{\"type\": \"master\", \"index\": 0}"
export TF_CONFIG="{\"cluster\":${config}, \"task\":${task}}"
run master


# Workers (Number of GPUS-1 one used by the master server)
for gpu in 0 1
do
    task="{\"type\": \"worker\", \"index\": $gpu}"
    export TF_CONFIG="{\"cluster\":${config}, \"task\":${task}}"
    export CUDA_VISIBLE_DEVICES="$gpu"

    run "worker${gpu}"
done

