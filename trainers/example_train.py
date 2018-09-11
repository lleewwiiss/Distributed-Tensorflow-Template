from base.trainer import BaseTrain
import tensorflow as tf
from models.example_model import Mnist
from data_loader.tfrecord_loader import TFRecordDataLoader


class ExampleTrainer(BaseTrain):
    def __init__(
        self,
        config: dict,
        model: Mnist,
        train: TFRecordDataLoader,
        val: TFRecordDataLoader,
        pred: TFRecordDataLoader,
    ) -> None:
        """
        This function will generally remain unchanged, it is used to train and export the model. The only part which
        may change is the run configuration, and possibly which execution to use (training, eval etc)
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
        run_config = tf.estimator.RunConfig(session_config=config)
        # set output directory
        run_config = run_config.replace(model_dir=self.config["job_dir"])

        # intialise the estimator with your model
        estimator = tf.estimator.Estimator(model_fn=self.model.model, config=run_config)

        # create train and eval specs for estimator, it will automatically convert the tf.Dataset into an input_fn
        train_spec = tf.estimator.TrainSpec(lambda: self.train.input_fn())

        # set the time between evaluations - throttle_secs
        eval_spec = tf.estimator.EvalSpec(
            lambda: self.val.input_fn(),
            steps=self.config["eval_steps"],
            throttle_secs=self.config["throttle_secs"],
        )

        # initialise a wrapper to do training and evaluation, this also handles exporting checkpoints/tensorboard info
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        # after training export the final model for use in tensorflow serving
        self._export_model(estimator, self.config["export_path"])

    def _export_model(
        self, estimator: tf.estimator.Estimator, save_location: str
    ) -> None:
        """
        Used to export your model in a format that can be used with
        Tf.Serving
        :param estimator: your estimator function
        """
        # this should match the input shape of your model
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
