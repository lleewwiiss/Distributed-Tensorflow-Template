from data_loader.data_loader import TFRecordDataLoader
from models.model import MNIST
from trainers.train import RawTrainer
from utils.utils import get_args
from utils import hyper_parameters
import tensorflow_datasets as tfds


def init() -> None:
    """
    The main function of the project used to initialise all the required classes
    used when training the model
    """
    # get input arguments
    args = get_args()
    # get hyper parameters
    params = hyper_parameters.HP
    # combine both into dictionary
    config = {**params, **args}

    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()
    # create your data generators for each mode
    train_data = TFRecordDataLoader(config, mode="train", mnist=mnist_builder)

    val_data = TFRecordDataLoader(config, mode="val", mnist=mnist_builder)

    test_data = TFRecordDataLoader(config, mode="test", mnist=mnist_builder)

    # initialise model
    model = MNIST(config)

    # initialise the estimator
    trainer = RawTrainer(config, model, train_data, val_data, test_data)

    # start training
    trainer.run()


if __name__ == "__main__":
    init()
