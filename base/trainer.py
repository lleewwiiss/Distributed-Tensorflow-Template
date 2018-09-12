import tensorflow as tf
from base import data_loader, model
from typing import Callable


class BaseTrain:
    def __init__(
        self,
        config: dict,
        model: model.BaseModel,
        train: data_loader.DataLoader,
        val: data_loader.DataLoader,
        pred: data_loader.DataLoader,
    ) -> None:
        """
        Simple framework to train, evaluate, test and export your model. The
        higher level TF API handles most of the work
        :param config: global configuration
        :param model: input function used to initialise model
        :param train: the training dataset
        :param val: the evaluation dataset
        :param pred: the prediction dataset
        """
        self.model = model
        self.config = config
        self.train = train
        self.val = val
        self.pred = pred

    def run(self) -> None:
        """
        Implement the required logic for training your model, the only changes
        will be configuration changes
        """
        raise NotImplementedError

    def _export_model(
        self, estimator: tf.estimator.Estimator, save_location: str
    ) -> None:
        """
        Used to export your model in a format that can be used with
        Tf.Serving
        :param estimator: your estimator function
        """
        raise NotImplementedError

    def _predict(self, estimator: tf.estimator.Estimator, pred_fn: Callable) -> list:
        """
        Function to yield prediction results from the model
        :param estimator: your estimator function
        :param pred_fn: input_fn associated with prediction dataset
        :return: a list containing a prediction for each batch in the dataset
        """
        raise NotImplementedError
