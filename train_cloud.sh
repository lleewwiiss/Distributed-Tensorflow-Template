#!/usr/bin/env bash
# UPDATE ALL VARIABLES HERE
prefix="example"
now=$(date +"%Y%m%d_%H_%M_%S")
JOB_NAME=${1-"${prefix}_${now}"}
# link to a bucket on gsp
GCS_BUCKET="gs://example-bucket/"
# Batch size
TRAIN_BATCH=32
EVAL_BATCH=32
# learning rate
LR="0.001"
# number of epochs
EPOCHS="100"
# locations locally or on the cloud for your files
TRAIN_FILES="${GCS_BUCKET}data/train.tfrecords"
EVAL_FILES="${GCS_BUCKET}data/val.tfrecords"
# END OF VARIABLES

gcloud ml-engine jobs submit training "${JOB_NAME}" \
    --job-dir "${GCS_BUCKET}BATCH-${BATCH}-LR-${LR}-${DATE}" \
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
