from base.trainer import BaseTrain
import tensorflow.keras as K
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

    def run(self) -> None:

        # train and evaluate model
        self._train_and_evaluate()

        # get results after training and exporting model
        self._predict()

        # after training export the final model for use in tensorflow serving
        self._export_model()

    def _export_model(self) -> None:
        tf.keras.experimental.export_saved_model(self.model, self.config["export_dir"])

    def _predict(self) -> list:
        """
        Function to yield prediction results from the model
        :return: a list containing a prediction for each batch in the dataset
        """
        pass

    def train_one_step(self, model, optimizer, x, y):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = self.compute_loss(y, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        self.compute_accuracy(y, logits)
        return loss

    def trainer(self, model, optimizer):
        step = 0
        loss = 0.0
        accuracy = 0.0
        for x, y in self.train.input_fn():
            step += 1
            loss = self.train_one_step(model, optimizer, x, y)
            if tf.equal(step % 10, 0):
                tf.print('Step', step, ': loss', loss, '; accuracy',
                         self.compute_accuracy.result())
        return step, loss, accuracy

    def _train_and_evaluate(self):
        checkpoint_dir = self.config["job_dir"]
        if os.path.isfile(os.path.join(checkpoint_dir, "checkpoint")):
            self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")
        self.model.save_weights(checkpoint_path.format(epoch=0))

        # Run evaluation every 10 epochs
        steps_per_epoch = len(self.train) / self.config['batch_size']

        if self.config['debug']:
            step, loss, accuracy = self.trainer(self.model, self.optimizer)
            print('Final step', step, ': loss', loss, '; accuracy', self.compute_accuracy.result())
        else:
            callbacks = [
                K.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path, save_weights_only=True,
                    verbose=1, period=5
                ),
                K.callbacks.TensorBoard(
                    log_dir=self.config["job_dir"],
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True,
                ),
            ]
            self.model.compile(
                optimizer=self.optimizer, loss=self.compute_loss,
                metrics=['accuracy'])

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
