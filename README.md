<h2 align="center">Distributed Tensorflow Project Template</h2>

<p align="center">
<a href="https://github.com/maxwellmri/Distributed-Tensorflow-Template/blob/master/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
A template for creating pragmatic Tensorflow models using the Dataset and Estimator APIs and allowing for easy distributed training. Based heavily off https://github.com/MrGemy95/Tensorflow-Project-Template

# Table Of Contents

-  [In a Nutshell](#in-a-nutshell)
-  [Training](#training)
    - [Training on CPU](#training-on-cpu)
    - [Training on GPU](#training-on-gpu)
    - [Distributed local training](#distributed-local-training)
    - [Distributed cloud training](#distributed-cloud-training)
-  [In Details](#in-details)
    -  [Folder structure](#folder-structure)
    -  [Main Components](#main-components)
        -  [Models](#models)
        -  [Trainer](#trainer)
        -  [Data Loader](#data-loader)
        -  [Configuration](#configuration)
        -  [Task](#task)
        -  [Project Settings](#project-settings)
 -  [Contributing](#contributing)

# In a Nutshell   
**There is a [MNIST CNN](models/example_model.py) example within the template, prefixed with ***example***, which shows how to create and use the template. Delete these files if you don't need them**

Look for ***#TODO*** tags in these classes as hints of what you need to update to create your model.  

In a nutshell here's how to use this template,**for example** to implement a VGG model you should do the following:
-  Update the [model](models/model.py) script by creating a class  called VGG and inherit from "BaseModel".

```python
class VGG16(BaseModel):
    def __init__(self, config: dict) -> None:
        """
        :param config: global configuration
        """
        super().__init__(config)

  ```
- Update the create model function in this script to build the graph for your model architecture.    
```python
def _create_model(x: tf.Tensor, is_training: bool) -> tf.Tensor:
    """
    :param x: input data
    :param is_training: flag if currently training
    :return: completely constructed model
    """
    pass

  ```

- Update the prediction and serving outputs for your model.   
```python
# TODO: update model predictions
predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits),
}

if mode == tf.estimator.ModeKeys.PREDICT:
    # TODO: update output during serving
    export_outputs = {
        "labels": tf.estimator.export.PredictOutput(
            {"label": predictions["classes"], "id": features["id"]}
        )
    }
    return tf.estimator.EstimatorSpec(
        mode, predictions=predictions, export_outputs=export_outputs
    )
  ```
  
 - Add any summaries you want to have on tensorboard for training and evaluation.
```python
# TODO: update summaries for tensorboard
tf.summary.scalar("loss", loss)
tf.summary.image("input", tf.reshape(x, [-1, 28, 28, 1]))

# if mode is evaluation
if mode == tf.estimator.ModeKeys.EVAL:
    # TODO: update evaluation metrics
    summaries_dict = {
        "val_accuracy": tf.metrics.accuracy(
            labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=summaries_dict
    )
  ```
 
  - Change your optimizer to suit your architecture, ensure if you are using Batch Normalisation that you are 
  using control dependencies (see [MNIST](models/example_model.py) example).   
```python
# TODO: update optimiser
optimizer = tf.train.AdamOptimizer(lr)
  ```
 
- Update then [train](trainers/train.py) script, and make sure to inherit from the "BaseTrain" class and use your new model and data loaders
```python
class VGGTrainer(BaseTrain):
    def __init__(
        self,
        config: dict,
        model: VGGModel,
        train: TFRecordDataLoader,
        val: TFRecordDataLoader,
        pred: TFRecordDataLoader,
    ) -> None:
        """
        This function will generally remain unchanged, it is used to train and
        export the model. The only part which may change is the run
        configuration, and possibly which execution to use (training, eval etc)
        :param config: global configuration
        :param model: input function used to initialise model
        :param train: the training dataset
        :param val: the evaluation dataset
        :param pred: the prediction dataset
        """
        super().__init__(config, model, train, val, pred)
```

- In the same script update the export function to match the inputs for your model.
```python
def _export_model(
        self, estimator: tf.estimator.Estimator, save_location: str
    ) -> None:
        """
        Used to export your model in a format that can be used with
        Tf.Serving
        :param estimator: your estimator function
        """
        # this should match the input shape of your model
        # TODO: update this to your input used in prediction/serving
        x1 = tf.feature_column.numeric_column(
            "input", shape=[self.config["batch_size"], 28, 28, 1]
        )
        # create a list in case you have more than one input
        feature_columns = [x1]
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec
        )
        # export the saved model
        estimator.export_savedmodel(save_location, export_input_fn)
```

- Update [data loader](data_loader/data_loader.py) to correctly load your input data, add or remove augmentation as needed. This example
is for tfrecords, it is possible to use other input data types (https://www.tensorflow.org/guide/datasets). Doing data loading and 
pre-processing on CPU helps reduce GPU bottlenecks.
```python
def _parse_example(
        self, example: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """
        Used to read in a single example from a tf record file and do any augmentations necessary
        :param example: the tfrecord for to read the data from
        :return: a parsed input example and its respective label
        """
        # do parsing on the cpu
        with tf.device("/cpu:0"):
            # define input shapes
            # TODO: update this for your data set
            features = {
                "image": tf.FixedLenFeature(shape=[28, 28, 1], dtype=tf.float32),
                "label": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            }
            example = tf.parse_single_example(example, features=features)

            if self.mode == "train":
                input_data = self._augment(example["image"])
            else:
                input_data = example["image"]

        return {"input": input_data}, example["label"]

```
- Update the [task](initialisers/task.py) script ensuring your new model and trainer are used. These scripts are used to initialise and train your model.
```python
def init() -> None:
    """
    The main function of the project used to initialise all the required functions for training the model
    """
    # get input arguments
    args = get_args()
    # get static config information
    config = process_config()
    # combine both into dictionary
    config = {**config, **args}

    # initialise model
    model = VGGModel(config)
    # create your data generators for each mode
    train_data = TFRecordDataLoader(config, mode="train")

    val_data = TFRecordDataLoader(config, mode="val")

    test_data = TFRecordDataLoader(config, mode="test")

    # initialise the estimator
    trainer = VGGTrainer(config, model, train_data, val_data, test_data)

    # start training
    trainer.run()
```
- Update the [utils](utils/utils.py) script adding any input arguments you need for your model, these will be added to the global config. 
For variables that are unlikely to change you can add them to the static config dictionary.
```python
def process_config() -> dict:
    """
    Add in any static configuration that is unlikely to change very often
    :return: a dictionary of static configuration data
    """
    config = {"exp_name": "example_model_train"}

    return config
```

# Training
In order to train your model there is a series of bash scripts which can train your model for several different
training environments. All of the local scripts will create log files.  

***training_log.md*** is created, or appended to, each time you run one of the scripts. You can use it as a scratch file to 
track your experiment details.  

The stdout of your model will be written files in ***runlogs/*** where each respective process will have a log. It also creates a ***.pid*** files which can be used to 
to kill the process if need be. An example of ***training_log.md*** is shown below:
```
Example Training Job 
Learning Rate: 0.001
Epochs: 100
Batch Size (train/eval): 512/ 512
Hypothesis:
Model will converge quickly
Results:
Model diverged even quicker
```
If you are not comfortable with vim or do not want to use this, you can remove it from scripts.

For each of the scripts you are going to need to update the hyper-parameters you
are wanting to use for this training run. Cloud based file paths won't work on windows. Add any additional input arguments that you have added to your model.
```bash
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
```

Training on CPU
--------------
[Train CPU](train_local_cpu.sh)  
This script will train the model without using any GPUs and you can optionally
specify a python environment to run the project from. 
- ### Usage
```bash
Usage: ./train_local_cpu.sh [ENV_NAME]
```

Training on GPU
--------------
[Train GPU](train_local_single.sh)  
This script will train the model using on specific GPU and you can optionally
specify a python environment to run the project from. It will also check to ensure
you have setup the CUDA environment variables. To find out GPU usage or which 
***GPU_ID*** to use you can run this in your terminal. 
```markdown
nvidia-smi
```
- ### Usage
```bash
Usage: ./train_local_single.sh <GPU_ID> [ENV_NAME]
```


Distributed local training
--------------
[Train distributed GPU](train_local_dist.sh)  
This script will allow you to simulate a distributed training environment locally on as many GPUs
as your machine has. In order to do this you must split the GPUs into workers, masters and parameter servers.
GPUs can be allocated to each of these types. You can also set this up directly in python (https://www.tensorflow.org/deploy/distributed). The script contains an example using 3 GPUs:
```markdown
config="
{
    \"master\": [\"localhost:27182\"],
    \"ps\": [\"localhost:27183\"],
    \"worker\": [
        \"localhost:27184\",
        \"localhost:27185\"
        ]
}, \"environment\": \"cloud\""
...
# ensure parameter server doesn't use any of the GPUs in this case
export CUDA_VISIBLE_DEVICES=""
# Parameter Server can be run on cpu
task="{\"type\": \"ps\", \"index\": 0}"
export TF_CONFIG="{\"cluster\":${config}, \"task\":${task}}"
run ps

# Master should be run on GPU as it runs the evaluation
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
```
This setup has 1 master, 1 parameter server, and two workers. The master is allocated one GPU and
the workers also have 1 GPU each. The parameter sever will be run on CPU. When defining new configurations
you have to ensure that the ports used in the ***config*** are not in use.
- ### Usage
```bash
Usage: ./train_local_dist.sh [ENV_NAME]
```

Distributed cloud training
--------------
[Train Cloud](train_cloud.sh)  
This script requires that you have Google Cloud SDK installed (https://cloud.google.com/sdk/install), and a Google Cloud Platform account
with access to ml-engine. Trial GCP accounts come with credit if you want to try this out. Training on the cloud does cost money, but it is very simple once setup. 

- ### Job config
    The [hptuning_config.yaml](hptuning_config.yaml) file will be used to specify the resources you are requesting for this job.
    You are able to scale this for your needs, it will behave the same as the local distributed training.  
    More information here: https://cloud.google.com/ml-engine/docs/tensorflow/using-gpus  
    See pricing here: https://cloud.google.com/ml-engine/docs/pricing

It is required that the data is stored on GCP somewhere in a bucket, and you also need to specify what bucket to
export your model and checkpoints to. Ensure that any additional packages your model needs are defined
in [setup.py](setup.py) and make sure you aren't specifying packages that are already part of ml-engine (https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list)
- ### Usage
```bash
Usage: ./train_cloud.sh
```

# In Details
Folder structure
--------------

```
├──  base
│    ├── data_loader.py  - this script contains the abstract class of the tfrecord data loader.
│    ├── model.py        - this script contains the abstract class of the model.
│    └── trainer.py      - this script contains the abstract class of the model trainer.
│
├──  data                - this folder contains any data your project may need.
│
├──  data_loader  
│    └── data_loader.py  - this script is responsible for all data handling.
│
├── initialisers        
│   └── task.py          - this script is used to start training model
│   
├──  models              
│    └── model.py        - this script is where your model is defined for each training phase.
│  
├──  trainers
│    └── train.py        - this script is where your estimator configuration is defined.
│ 
└── utils
     ├── make_tfrec.py   - this script is an example how to create tfrecords from numpy or cvs files
     └── utils.py        - this script handles your input variables and defines a global config.

```

## Main Components

### Models

--------------
- #### **Base model**
    
    Base model is an abstract class that must be Inherited by any model you create, the only requirement is that 
    you implement a model function which is compliant with Estimator API.
    - ***Model*** This function to is where your experiment is defined
    - ***Create Model*** This function to is where your model architecture is defined

- #### **Your model**
    Here's where you implement your model.
    So you should:
    - Create your model class and inherit the base_model class
    - Override "model" where you write the tensorflow Estimator experiment
    - Override "create_model" where you write your model architecture

### Trainer

--------------
- #### **Base trainer**
    Base trainer is an abstract class that just wraps the training process.
    - ***Run*** This function sets up the Estimator configuration for the different training stages and runs your 
    training loop
    - ***Export Model*** This function exports the model to a given location with compatibility for Tensorflow Serving
    - ***Predict*** This function takes a prediction dataset and returns predicted values using your model based off it's last
    saved checkpoint
    
- #### **Your trainer**
     Here's what you should implement in your trainer.  
     So you should:
     - Override the export_model function to match the inputs of your model
     - Override the predict function for your projects requirements

### Data Loader

--------------
This class is responsible for all data handling and processing and provides an easy interface that can be used by the trainer.
The current loader uses tfrecords which are the recommended way of loading data into a Tensorflow model.  
If you are using tfrecords you should:
- Update the [parse_example](data_loader/data_loader.py) function so the input feature maps are the same as your model
- Add or remove any augmentation methods
 
### Configuration

--------------
Add any static configuration variables to the dict in [utils](utils/utils.py), otherwise it is recommended to use input variables
to handle the configuration settings

### Task

--------------
Here's where you combine all previous part.
1. Update the model to use your new model class
2. Update the datasets to point to your data loader function for each respective training step (train, eval, prediction etc) 
3. Update the trainer to use your new trainer class

This can now be used as the entry point to run your experiment.

### Project Settings
- ### Flake8
    This project includes settings for the python linter Flake8 which are compatible with black and mypy. Update [.flake8](.flake8) 
    if you would like to change these settings
- ### Black
    This project includes settings for the python formatter Black. The settings for black should be defined
    in the [pyproject.toml](pyproject.toml) file
- ### MyPy
    This project includes settings for the python optional static type check Mypy. The settings are
    defined in [mypy.ini](mypy.ini) you can define files to ignore in here.
- ### Pre-commit
    This project includes settings for pre-commit hooks to run flake8, black and mypy. The settings are
    defined in [.pre-commit-config.yaml](.pre-commit-config.yaml). To maintain good code quality it is recommended you install
    and use pre-commit, which runs these tools each time you commit, ensuring they pass before you can push.
    If you are wanting to use this, make sure the other tools are installed using pip then run within your model folder:
```bash
pip install pre-commit
pre-commit install
```
- ### Setup
    Update the [setup.py](setup.py) details and [requirements.txt](requirements.txt) for your project, specifying any packages that you
    need for your project.

# Contributing

--------------
Any kind of enhancement or contribution is welcomed.
