from data_loader.tfrecord_loader import TFRecordDataLoader
from models.example_model import Mnist
from trainers.example_train import ExampleTrainer
from utils.utils import get_args, process_config


def init() -> None:
    """
    The main function of the project used to initialise all the required functions for training the model
    """
    # get input arguments
    args = get_args()
    # get static config information
    config = process_config()
    # combine both into dictionary
    config = {**config, **args}

    # initialise model
    model = Mnist(config)
    # create your data generators for each mode
    train_data = TFRecordDataLoader(config, mode="train")

    val_data = TFRecordDataLoader(config, mode="val")

    test_data = TFRecordDataLoader(config, mode="test")

    # initialise the estimator
    trainer = ExampleTrainer(config, model, train_data, val_data, test_data)

    # start training
    trainer.run()


if __name__ == "__main__":
    init()
