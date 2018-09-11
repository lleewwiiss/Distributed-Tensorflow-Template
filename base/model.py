import tensorflow as tf
from typing import Dict


class BaseModel:
    def __init__(self, config: dict) -> None:
        """
        All models will follow the same structure, the only difference will be the architecture and the evaluation
        metrics can be defined on a use case basis
        :param config: global configuration
        """
        self.config = config

    def model(
        self, features: Dict[str, tf.Tensor], labels: tf.Tensor, mode: str
    ) -> tf.estimator.EstimatorSpec:
        """
        Implement the logic of your model, including any metrics to track on tensorboard and the architecture of your
        network
        :param features: A dictionary of potential inputs for your model
        :param labels: Input label set
        :param mode: Current training mode (train, test, predict)
        """
        raise NotImplementedError
