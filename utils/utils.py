import argparse


def get_args() -> dict:
    """
    Get command line arguments add and remove any needed by your project
    :return: Namespace of command arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--debug",
        help="flag for eager mode",
        required=True,
    )

    parser.add_argument(
        "--train-files",
        help="GCS or local paths to training data",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--export-path",
        type=str,
        help="Where to export the saved model to locally or on GCP", required=True
    )

    parser.add_argument(
        "--eval-files",
        help="GCS or local paths to evaluation data",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--test-files", help="GCS or local paths to test data", nargs="+", required=True
    )
    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints and export models",
        required=True,
    )

    args, unknown = parser.parse_known_args()

    return vars(args)
