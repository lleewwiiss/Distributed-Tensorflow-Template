from base.data_loader import DataLoader
import tensorflow as tf
import multiprocessing
from typing import Tuple, Dict
import os


class TFRecordDataLoader(DataLoader):
    def __init__(self, config: dict, mode: str) -> None:
        """
        An example of how to create a dataset using tfrecords inputs
        :param config: global configuration
        :param mode: current training mode (train, test, predict)
        """
        super().__init__(config, mode)

        if self.mode == "train":
            self.file_names = [
                os.path.join(os.getcwd(), x) for x in self.config["train_files"]
            ]
            self.batch_size = self.config["train_batch_size"]
        elif self.mode == "val":
            self.file_names = [
                os.path.join(os.getcwd(), x) for x in self.config["eval_files"]
            ]
            self.batch_size = self.config["eval_batch_size"]
        else:
            self.file_names = [
                os.path.join(os.getcwd(), x) for x in self.config["test_files"]
            ]

    def input_fn(self) -> tf.data.Dataset:
        """
        Create a tf.Dataset using tfrecords as inputs, use parallel loading and augmentation using the CPU to
        reduce bottle necking of operations on the GPU
        :return: a Dataset function
        """
        dataset = tf.data.TFRecordDataset(self.file_names)
        # create a parallel parsing function based on number of cpu cores
        dataset = dataset.map(
            map_func=self._parse_example, num_parallel_calls=multiprocessing.cpu_count()
        )

        # only shuffle training data
        if self.mode == "train":
            # shuffles and repeats a Dataset returning a new permutation for each epoch. with serialise compatibility
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(
                    buffer_size=self.batch_size * 10, count=self.config["num_epochs"]
                )
            )
        else:
            dataset = dataset.repeat(self.config["num_epochs"])
        # create batches of data
        dataset = dataset.batch(batch_size=self.batch_size)
        return dataset

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

    @staticmethod
    def _augment(example: tf.Tensor) -> tf.Tensor:
        """
        Randomly augment the input image to try improve training variance
        :param example: parsed input example
        :return: the same input example but possibly augmented
        """
        return tf.image.random_flip_left_right(example)
