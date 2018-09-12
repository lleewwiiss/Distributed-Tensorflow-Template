from base.data_loader import DataLoader
import tensorflow as tf
import multiprocessing
from typing import Tuple, Dict
import random


class TFRecordDataLoader(DataLoader):
    def __init__(self, config: dict, mode: str) -> None:
        """
        An example of how to create a dataset using tfrecords inputs
        :param config: global configuration
        :param mode: current training mode (train, test, predict)
        """
        super().__init__(config, mode)

        # Get a list of files in case you are using multiple tfrecords
        if self.mode == "train":
            self.file_names = self.config["train_files"]
            self.batch_size = self.config["train_batch_size"]
        elif self.mode == "val":
            self.file_names = self.config["eval_files"]
            self.batch_size = self.config["eval_batch_size"]
        else:
            self.file_names = self.config["test_files"]

    def input_fn(self) -> tf.data.Dataset:
        """
        Create a tf.Dataset using tfrecords as inputs, use parallel
        loading and augmentation using the CPU to
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
            # shuffles and repeats a Dataset returning a new permutation for each epoch. with serialised compatibility
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(
                    buffer_size=len(self) // self.config["train_batch_size"]
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
            # TODO: update this for your data set
            features = {
                "image": tf.FixedLenFeature(shape=[28, 28, 1], dtype=tf.float32),
                "label": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            }
            example = tf.parse_single_example(example, features=features)

            # only augment training data
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
        # random rotation
        if random.uniform(0, 1) > 0.5:
            example = tf.contrib.image.rotate(
                example, tf.random_uniform((), minval=-0.2, maxval=0.2)
            )
        # random noise
        if random.uniform(0, 1) > 0.5:
            # assumes values are normalised between 0 and 1
            noise = tf.random_normal(
                shape=tf.shape(example), mean=0.0, stddev=0.2, dtype=tf.float32
            )
            example = example + noise
            example = tf.clip_by_value(example, 0.0, 1.0)
            # random flip
            example = tf.image.random_flip_up_down(example)
        return tf.image.random_flip_left_right(example)

    def __len__(self) -> int:
        """
        Get number of records in the dataset
        :return: number of samples in all tfrecord files
        """
        return sum(
            1 for fn in self.file_names for _ in tf.python_io.tf_record_iterator(fn)
        )
