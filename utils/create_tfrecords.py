import tensorflow as tf
import glob
import numpy as np
from random import shuffle
import sys
import pandas as pd


# download the csvs from https://www.kaggle.com/c/digit-recognizer/data and place in data dir
def main() -> None:
    """
    Convert a set of numpy files to a set of train, eval and test tfrecord files. This example is for MNIST
    """
    data_dir = "../data/temp/"
    # get all images
    filenames = glob.glob(f"{data_dir}*_image.npy")
    shuffle(filenames)

    train_files = filenames[0 : int(0.8 * len(filenames))]

    val_files = filenames[int(0.8 * len(filenames)) :]

    print("Creating training file")
    filename = "{}train.tfrecords".format(
        "../data/"
    )  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    for i in train_files:
        # Load the image
        try:
            img = np.load(i)
            label = np.load(i.replace("_image", "_label"))
        except UnicodeError as e:
            print(i)

        # Create a feature
        feature = {
            "image": tf.train.Feature(
                float_list=tf.train.FloatList(value=img.flatten().tolist())
            ),
            "label": tf.train.Feature(
                int64_list=tf.train.Int64List(value=label.flatten().tolist())
            ),
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    print("Creating validation file")

    filename = "{}val.tfrecords".format(
        "../data/"
    )  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    for i in val_files:
        # Load the image
        try:
            img = np.load(i)
            label = np.load(i.replace("_image", "_label"))
        except UnicodeError as e:
            print(i)

        # Create a feature
        feature = {
            "image": tf.train.Feature(
                float_list=tf.train.FloatList(value=img.flatten().tolist())
            ),
            "label": tf.train.Feature(
                int64_list=tf.train.Int64List(value=label.flatten().tolist())
            ),
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    filenames = glob.glob(f"{data_dir}*_test.npy")

    print("Creating test file")
    filename = "{}test.tfrecords".format(
        "../data/"
    )  # address to save the TFRecords file
    # open the TFRecords file

    filenames = sorted(filenames)
    writer = tf.python_io.TFRecordWriter(filename)
    for i in filenames:
        # Load the image
        img = np.load(i)

        # Create a feature
        feature = {
            "image": tf.train.Feature(
                float_list=tf.train.FloatList(value=img.flatten().tolist())
            )
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    print("Finished converting files")


def _extract_data() -> None:
    """
    Convert the MNIST csv files to individual npy files and respective label files
    """
    df = pd.read_csv("train.csv")

    for j, row in df.iterrows():
        curr_row = row.tolist()
        label = [curr_row[0]]
        rows = [np.array(curr_row[i : i + 28]) for i in range(2, len(curr_row), 28)]
        np.save("temp/{}_label.npy".format(j), np.array(label))
        np.save("temp/{}_image.npy".format(j), np.array(rows))

    df = pd.read_csv("test.csv")

    for j, row in df.iterrows():
        curr_row = row.tolist()
        rows = [np.array(curr_row[i : i + 28]) for i in range(1, len(curr_row), 28)]
        np.save("temp/{}_test.npy".format(j), np.array(rows))


if __name__ == "__main__":
    if True:
        _extract_data()
    main()
