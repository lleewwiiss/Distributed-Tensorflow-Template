#!/usr/bin/env bash
prefix="example-train"
now=$(date +"%Y%m%d_%H_%M_%S")
JOB_NAME=${1-"${prefix}_${now}"}

GCS_BUCKET="gs://example-bucket/"

# Batch size
BATCH=4
# learning rate
LR="0.1"
# number of epochs
EPOCHS="100"

TRAIN_FILES="${GCS_BUCKET}data/train.tfrecords"
EVAL_FILES="${GCS_BUCKET}data/val.tfrecords"

gcloud ml-engine jobs submit training "$JOB_NAME" \
    --job-dir "$GCS_BUCKETBATCH-$BATCH-LR-$LR" \
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
