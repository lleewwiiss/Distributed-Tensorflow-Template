import tensorflow as tf
from typing import Tuple, Dict, Sized


class DataLoader(Sized):
    """
    The parent class of your data loaders, all data loaders should implement an input_function which creates a
    tf.Dataset and also a parsing function, which is used to read in data. It is recommended
    that pre-processing is done online using cpu but this is not required
    """

    def __init__(self, config: dict, mode: str) -> None:
        """
        The Dataset will be dependent on the mode (train, eval etc)
        :param config: global configuration settings
        :param mode: current training mode (train, test, predict)
        """
        self.config = config
        self.mode = mode

    def input_fn(self) -> tf.data.Dataset:
        """
        Create a dataset which reads in some data source (e.g. tfrecords, csv etc)
        """
        raise NotImplementedError

    def _parse_example(
        self, example: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """
        A function for parsing single serialised Examples. To be used when
        input files are TFRecords.

        :param example: the location of the data to parse the example from
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Get number of records in the dataset
        See example_train.py for example implementation
        :return: number of samples in all tfrecord files
        """
        raise NotImplementedError
