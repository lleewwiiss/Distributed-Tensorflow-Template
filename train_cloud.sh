#!/usr/bin/env bash
##########################################################
# where to write tfevents
GCS_BUCKET="gs://model-exports"
# experiment settings
BATCH=512
LR=0.001
EPOCHS=100
# create a job name for the this run
prefix="example"
now=$(date +"%Y%m%d_%H_%M_%S")
JOB_NAME="$prefix"_"$now"
# locations locally or on the cloud for your files
TRAIN_FILES="${GCS_BUCKET}data/train.tfrecords"
EVAL_FILES="${GCS_BUCKET}data/val.tfrecords"
##########################################################

gcloud ml-engine jobs submit training "${JOB_NAME}" \
    --job-dir "${GCS_BUCKET}BATCH-${BATCH}-LR-${LR}-${now}" \
    --package-path initialisers \
    --module-name "initialisers.task" \
    --region us-central1 \
    --runtime-version 1.8 \
    --config hptuning_config.yaml \
    -- \
    --job-dir ${GCS_BUCKET} \
    --train-batch-size ${BATCH} \
    --eval-batch-size ${BATCH} \
    --learning-rate ${LR} \
    --num-epochs ${EPOCHS} \
    --train-files ${TRAIN_FILES} \
    --eval-files ${EVAL_FILES} \
    --export-path "${GCS_BUCKET}exports"
