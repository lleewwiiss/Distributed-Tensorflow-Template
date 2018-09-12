import tensorflow as tf
from base.model import BaseModel
from typing import Dict


class RawModel(BaseModel):
    def __init__(self, config: dict) -> None:
        """
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
        x = features["input"]

        # TODO: create graph
        # initialise model architecture
        logits = self._create_model(x, is_training)

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

        # calculate loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # TODO: update summaries for tensorboard
        tf.summary.scalar("loss", loss)
        tf.summary.image("input", tf.reshape(x, [-1, 28, 28, 1]))

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

        # assert only reach this point during training
        assert mode == tf.estimator.ModeKeys.TRAIN

        # create learning rate variable for hyper param tuning
        lr = tf.Variable(
            initial_value=self.config["learning_rate"], name="learning-rate"
        )

        # TODO: update optimiser
        optimizer = tf.train.AdamOptimizer(lr)

        train_op = optimizer.minimize(
            loss,
            global_step=tf.train.get_global_step(),
            colocate_gradients_with_ops=True,
        )

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    @staticmethod
    def _create_model(x: tf.Tensor, is_training: bool) -> tf.Tensor:
        """
        Implement the architecture of your model
        :param x: input data
        :param is_training: flag if currently training
        :return: completely constructed model
        """
        pass
