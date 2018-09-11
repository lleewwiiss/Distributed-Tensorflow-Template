#!/bin/bash
if [ $# -ne 1 ]; then
    echo $0: usage: train_local_dist envname
    exit 1
fi

# ensure you are linked to cuda on the machine
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda

# needed to use virtualenvs
set -euo pipefail

# create a job name for the this run
prefix="example"
now=$(date +"%Y%m%d_%H_%M_%S")
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
        --train-batch-size ${BATCH} \
        --eval-batch-size ${BATCH} \
        --learning-rate ${LR} \
        --num-epochs ${EPOCHS} \
        --train-files ${TRAIN_FILES} \
        --eval-files ${EVAL_FILES} \
        --test-files ${TEST_FILES} \
        --export-path "${GCP_BUCKET}exports" \
               &>runlogs/$1.log &
               echo "$!" > runlogs/$1.pid

}

# activate the virtual environment
set +u
source $1/bin/activate
set -u

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

