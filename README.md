A template for tensorflow models: Any question ask Lewis.
Submit pull request if you have ideas. This is based on python3.5 and uses tensorflow-gpu 1.8

## Usage:  
The only classes you need the change:  
# models.model  
Implement your model logic in this module you probably will only need to change the parts with the TODO tags
# data_loaded.tfrecord_loader  
Update the feature dictionary to match the inputs of your model and add any augmentation methods you need. This assumes
that all data has already been preprocessed as this is the assumption made for inference. Add any addition input data 
that is required for your model.
# utils.utils  
Update the config and command-line and static arguments for your use case  
# hptuning_config.yaml  
This is used to allocate the ml-engine training worker, change based on your needs, add any parameters you want to tune.
You do not need to tune parameters more info (https://cloud.google.com/ml-engine/docs/tensorflow/machine-types)
# requirements.txt  
Ensure this is up to date and does not include any redundant packages. Try minimise other packages aside from tensorflow
# setup.py  
Update this for your project and your packages, this is used in GCP when creating a job in ML engine. Ensure you don't 
have packages in here that are already listed in your runtime version 
(https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list)  

## Running training:
Local logs will be exported to runlogs directory which gets created when the scripts are run, and exports will be saved
into the local jobs directory.
# train_cloud.sh 
Usage: train_cloud  
This script will start a training job in ML engine update it for your use case and specific input paprams  
# train_local_dist.sh  
Usage: train_local_dist envname  
This will simulate an ml-engine job locally and will allow for gpu parallisation, this script will need to 
be updated based on your machine, the example is for bruce. This requires there to be a virtual environment to be created 
within the the project e.g. template/train-example  
# train_local_single.sh  
Usage: train_local_single envname gpu_id  
Run a normal training session using 1 gpu 