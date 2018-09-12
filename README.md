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
In a nutshell here's how to use this template,**for example** to implement a VGG model you should do the following:
-  In the [models](models) folder update the template class to VGG and inherit from "BaseModel".

```python
class VGG16(BaseModel):
    def __init__(self, config: dict) -> None:
        """
        :param config: global configuration
        """
        super().__init__(config)

  ```
- Update the create model function to build the graph for your model architecture.    
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

# if mode is prediction
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
  using control dependencies (see mnist example).   
```python
# TODO: update optimiser
optimizer = tf.train.AdamOptimizer(lr)
  ```
 
- In the [trainers](trainers) folder, update the trainer script and make sure to inherit from the "BaseTrain" class.
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

- In data loader update the data loader to correctly load your input data, add or remove augmentation as needed.
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
- In your initialiser ensure your new model and trainer are used. These scripts are used to initialise and train your model.
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
- Within [utils](utils/utils.py) add any input arguments you need for your model, these will be added to the global config. 
For variables that are unlikely to change you can add them to the config dictionary.
```python
def process_config() -> dict:
    """
    Add in any static configuration that is unlikely to change very often
    :return: a dictionary of static configuration data
    """
    config = {"exp_name": "example_model_train"}

    return config
```
**There is a MNIST CNN example within the template, which shows how to create and use the mode. Delete these files if you don'e need them**

# Training
Training on CPU
--------------

Training on GPU
--------------

Distributed local training
--------------

Distributed cloud training
--------------

# In Details
Folder structure
--------------

```
├──  base
│    ├── data_loader.py  - this function contains the abstract class of the tfrecord data loader.
│    ├── model.py        - this function contains the abstract class of the model.
│    └── trainer.py      - this function contains the abstract class of the model trainer.
│
├──  data                - this folder contains any data your project may need.
│
├──  data_loader  
│    └── data_loader.py  - this function is responsible for all data handling.
│
├── initialisers        
│   └── task.py          - this function is used to start training model
│   
├──  models              
│    └── model.py        - this function is where your model is defined for each training phase.
│  
├──  trainers
│    └── train.py        - this function is where your estimator configuration is defined.
│ 
└── utils
     ├── create_tfrecords.py
     └── utils.py        - this function handles your input variables and defines a global config.

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
    - Override "create model" where you write your model architecture

### Trainer

--------------
- #### **Base trainer**
    Base trainer is an abstract class that just wrap the training process.
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
This class is responsible for all data handling and processing and provide an easy interface that can be used by the trainer.
The current loader uses tfrecords which are the recommended way of loading data into a Tensorflow model.  
If you are using tfrecords you should:
- Update the parse_example function so the input feature maps are the same as your model
- Add or remove any augmentation methods
 
### Configuration

--------------
Add any static configuration variables to the dict in utils, otherwise it is recommended to use input variables
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
    This project includes settings for the python linter Flake8 which are compatible with black and mypy. Update ***.flake8*** 
    if you would like to change these settings
- ### Black
    This project includes settings for the python formatter Black. The settings for black should be defined
    in the ***pyproject.toml*** file
- ### MyPy
    This project includes settings for the python option static type check Mypy. The settings are
    defined in ***mypy.ini*** you can define files to ignore in here.
- ### Pre-commit
    This project includes settings for pre-commit hooks to run flake8, black and mypy. The settings are
    defined in ***.pre-commit-config.yaml***. To maintain good code quality it is recommended to install
    and use pre-commit, which runs these tools each time you commit, ensuring they pass before you can push.
    If you are wanting to use this, make sure the other tools are installed using pip then run:
```bash
pip install pre-commit
pre-commit install
```
- ### Setup
    Update the ***setup.py*** and ***requirements.txt*** for your project, specifying any packages that you
    need for your project.

# Contributing

--------------
Any kind of enhancement or contribution is welcomed.