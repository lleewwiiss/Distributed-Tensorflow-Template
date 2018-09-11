import argparse
import os
import tensorflow as tf


def process_config() -> dict:
    """
    Add in any static configuration that is unlikely to change very often
    :return: a dictionary of static configuration data
    """
    config = {"exp_name": "example_model_train"}

    return config


def get_args() -> dict:
    """
    Get command line arguments add and remove any needed by your project
    :return: Namespace of command arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-files",
        help="GCS or local paths to training data",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--num-epochs",
        help="""\
        Maximum number of training data epochs on which to train.
        If both --max-steps and --num-epochs are specified,
        the training job will run for --max-steps or --num-epochs,
        whichever occurs first. If unspecified will run for --max-steps.\
        """,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--train-batch-size", help="Batch size for training steps", type=int, default=40
    )
    parser.add_argument(
        "--throttle-secs",
        help="""Do not re-evaluate unless the last evaluation was started at least
        this many seconds ago. Of course, evaluation does not occur if no new
        checkpoints are available, hence, this is the minimum.s""",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--eval-batch-size",
        help="Batch size for evaluation steps",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--export-path", type=str, help="""Where to export the saved model to"""
    )

    parser.add_argument(
        "--eval-files",
        help="GCS or local paths to evaluation data",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--test-files",
        help="GCS or local paths to evaluation data",
        nargs="+",
        required=True,
    )
    # Training arguments
    parser.add_argument(
        "--learning-rate",
        help="Learning rate for the optimizer",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints and export models",
        required=True,
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="DEBUG",
        help="Set logging verbosity",
    )
    # Experiment arguments
    parser.add_argument(
        "--eval-delay-secs",
        help="How long to wait before running first evaluation",
        default=180,
        type=int,
    )
    parser.add_argument(
        "--keep-prob", help="keep probability for dropout", default=0.5, type=int
    )
    parser.add_argument(
        "--min-eval-frequency",
        help="Minimum number of training steps between evaluations",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--max-steps",
        help="""\
        Steps to run the training job for. If --num-epochs is not specified,
        this must be. Otherwise the training job will run indefinitely.\
        """,
        type=int,
    )
    parser.add_argument(
        "--eval-steps",
        help="""\
        Number of steps to run evaluation for at each checkpoint.
        If unspecified will run until the input from --eval-files is exhausted
        """,
        default=None,
        type=int,
    )
    parser.add_argument(
        "--export-format",
        help="The input format of the exported SavedModel binary",
        choices=["JSON", "CSV", "EXAMPLE"],
        default="EXAMPLE",
    )

    args, unknown = parser.parse_known_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)

    # Set C++ Graph Execution level verbosity
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(tf.logging.__dict__[args.verbosity] / 10)

    if unknown:
        tf.logging.warn("Unknown arguments: {}".format(unknown))

    return vars(args)
