from base.trainer import BaseTrain
import tensorflow as tf
from models.model import RawModel
from data_loader.data_loader import TFRecordDataLoader
from typing import Callable


class RawTrainer(BaseTrain):
    def __init__(
        self,
        config: dict,
        model: RawModel,
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

    def run(self) -> None:
        # allow memory usage to me scaled based on usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # get number of steps required for one pass of data
        steps_pre_epoch = len(self.train) / self.config["train_batch_size"]
        # save_checkpoints_steps is number of batches before eval
        run_config = tf.estimator.RunConfig(
            session_config=config,
            save_checkpoints_steps=steps_pre_epoch
            * 10,  # number of batches before eval/checkpoint
            log_step_count_steps=steps_pre_epoch,  # number of steps in epoch
        )
        # set output directory
        run_config = run_config.replace(model_dir=self.config["job_dir"])

        # intialise the estimator with your model
        estimator = tf.estimator.Estimator(model_fn=self.model.model, config=run_config)

        # create train and eval specs for estimator, it will automatically convert the tf.Dataset into an input_fn
        train_spec = tf.estimator.TrainSpec(
            lambda: self.train.input_fn(),
            max_steps=self.config["num_epochs"] * steps_pre_epoch,
        )

        eval_spec = tf.estimator.EvalSpec(lambda: self.val.input_fn())

        # initialise a wrapper to do training and evaluation, this also handles exporting checkpoints/tensorboard info
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        # after training export the final model for use in tensorflow serving
        self._export_model(estimator, self.config["export_path"])

        # get results after training and exporting model
        self._predict(estimator, self.pred.input_fn)

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

    def _predict(self, estimator: tf.estimator.Estimator, pred_fn: Callable) -> list:
        """
        Function to yield prediction results from the model
        :param estimator: your estimator function
        :param pred_fn: input_fn associated with prediction dataset
        :return: a list containing a prediction for each batch in the dataset
        """
        pass
