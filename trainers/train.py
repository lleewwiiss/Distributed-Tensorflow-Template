from base.trainer import BaseTrain
import tensorflow.keras as K
from comet_ml import Experiment
from models.model import MNIST
from data_loader.data_loader import TFRecordDataLoader
import tensorflow as tf
import os


class RawTrainer(BaseTrain):
    def __init__(
        self,
        config: dict,
        model: MNIST,
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
        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.optimizer = K.optimizers.Adam()
        self.compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.experiment = Experiment(
            api_key="NAoCVwaFTcgOltYJJs1wimcnZ",
            project_name="template",
            workspace="maxwell",
            auto_param_logging=False)
        self.experiment.log_paraters(self.config)

    def run(self) -> None:

        # train and evaluate model
        self._train_and_evaluate()

        # get results after training and exporting model
        self._predict()

        # after training export the final model for use in tensorflow serving
        self._export_model()

    def _export_model(self) -> None:
        tf.keras.experimental.export_saved_model(
            model=self.model,
            saved_model_path=self.config["export_dir"],
            input_signature=None,
            serving_only=True
        )

    def _predict(self) -> list:
        """
        Function to yield prediction results from the model
        :return: a list containing a prediction for each batch in the dataset
        """
        with self.experiment.test():
            pass

    def _train_and_evaluate(self):
        checkpoint_dir = self.config["job_dir"]
        if os.path.isfile(os.path.join(checkpoint_dir, "checkpoint")):
            self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")
        self.model.save_weights(checkpoint_path.format(epoch=0))

        # Run evaluation every 10 epochs
        steps_per_epoch = len(self.train) / self.config['batch_size']

        callbacks = [
            K.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, save_weights_only=True,
                verbose=1, period=5
            ),
            K.callbacks.TensorBoard(
                log_dir=self.config["job_dir"],
                histogram_freq=steps_per_epoch,
                write_graph=True,
                write_images=True,
            ),
        ]
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.compute_loss,
            metrics=['accuracy'],
            run_eagerly=self.config['debug'])

        with self.experiment.train():
            history = self.model.fit(
                x=self.train.input_fn(),
                epochs=self.config["train_epochs"],
                verbose=1,
                callbacks=callbacks,
                validation_data=self.val.input_fn(),
                initial_epoch=0,
                validation_freq=10,
                steps_per_epoch=steps_per_epoch
            )
