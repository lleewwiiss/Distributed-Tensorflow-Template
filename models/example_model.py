import tensorflow as tf
from base.model import BaseModel
from typing import Dict


class Mnist(BaseModel):
    def __init__(self, config: dict) -> None:
        """
        Create a model used to classify hand written images using the MNIST dataset
        :param config: global configuration
        """
        super().__init__(config)

    def model(
        self, features: Dict[str, tf.Tensor], labels: tf.Tensor, mode: str
    ) -> tf.Tensor:
        """
        Define your model metrics and architecture, the logic is dependent on the mode.
        :param features: A dictionary of potential inputs for your model
        :param labels: Input label set
        :param mode: Current training mode (train, test, predict)
        :return: An estimator spec used by the higher level API
        """
        # set flag if the model is currently training
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # get input data
        image = features["input"]
        # initialise model architecture
        logits = _create_model(image, self.config["keep_prob"], is_training)

        # define model predictions
        predictions = {
            "class": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits),
        }

        # if mode is prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            # define what to output during serving
            export_outputs = {
                "labels": tf.estimator.export.PredictOutput(
                    {"id": features["id"], "label": predictions["class"]}
                )
            }
            return tf.estimator.EstimatorSpec(
                mode, predictions=predictions, export_outputs=export_outputs
            )

        # calculate loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # add summaries for tensorboard
        tf.summary.scalar("loss", loss)
        tf.summary.image("input", tf.reshape(image, [-1, 28, 28, 1]))

        # if mode is evaluation
        if mode == tf.estimator.ModeKeys.EVAL:
            # create a evaluation metric
            summaries_dict = {
                "val_accuracy": tf.metrics.accuracy(
                    labels, predictions=predictions["classes"]
                )
            }
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=summaries_dict
            )

        # assert only reach this point during training
        assert mode == tf.estimator.ModeKeys.TRAIN

        # collect operations which need updating before back-prob e.g. Batch norm
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # create learning rate variable for hyper-parameter tuning
        lr = tf.Variable(
            initial_value=self.config["learning_rate"], name="learning-rate"
        )

        # initialise optimiser
        optimizer = tf.train.AdamOptimizer(lr)

        # Do these operations after updating the extra ops due to BatchNorm
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(
                loss,
                global_step=tf.train.get_global_step(),
                colocate_gradients_with_ops=True,
            )

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def _fc_block(x: tf.Tensor, size: int, is_training: bool, drop: float) -> tf.Tensor:
    """
    Create a fully connected block using batch-norm and drop out
    :param x: input layer which proceeds this block
    :param size: number of nodes in layer
    :param is_training: flag if currently training
    :param drop: percentage of data to drop
    :return: fully connected block of layers
    """
    x = tf.layers.Dense(size)(x)
    x = tf.layers.BatchNormalization(fused=True)(x, training=is_training)
    x = tf.nn.relu(x)
    return tf.layers.Dropout(drop)(x, training=is_training)


def _conv_block(
    x: tf.Tensor, layers: int, filters: int, is_training: bool
) -> tf.Tensor:
    """
    Create a convolutional block using batch norm
    :param x: input layer which proceeds this block
    :param layers: number of conv blocks to create
    :param filters: number of filters in each conv layer
    :param is_training: flag if currently training
    :return: block/s of residual layers
    """
    for i in range(layers):
        x = tf.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.layers.BatchNormalization(fused=True)(x, training=is_training)
        x = tf.nn.relu(x)
    return tf.layers.MaxPooling2D(2, 2, padding="valid")(x)


def _create_model(x: tf.Tensor, drop: float, is_training: bool) -> tf.Tensor:
    """
    A basic deep CNN used to train the MNIST classifier
    :param x: input data
    :param drop: percentage of data to drop during dropout
    :param is_training: flag if currently training
    :return: completely constructed model
    """
    x = tf.reshape(x, [-1, 28, 28, 1])
    _layers = [1, 1]
    _filters = [32, 64]

    # create the residual blocks
    for i, l in enumerate(_layers):
        x = _conv_block(x, l, _filters[i], is_training)

    x = tf.layers.Flatten()(x)
    _fc_size = [1024]

    # create the fully connected blocks
    for s in _fc_size:
        x = _fc_block(x, s, is_training, drop)
    # add an output layer (10 classes, one output for each)
    return tf.layers.Dense(10)(x)
