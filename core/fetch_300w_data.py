#%matplotlib inline
from matplotlib import image
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from PIL import Image, ImageOps
import numpy as np
import random

from params.project_config import data_folder

def import_300lw_data():
    print("Imports Collected")

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.config.list_physical_devices('GPU')


    ds_train, ds_info = tfds.load(
        'the300w_lp',
        split=['train'],
        shuffle_files=True,
        with_info=True,
    )
    print("Loaded")


    print(type(ds_train))
    print(type(ds_train[0]))
    #print(ds_info)
    print(len(ds_train[0]))


    num_entries = 61226
    percent = int(num_entries / 100)
    train_percent = 80
    out_size = (256, 256)
    print("Starting export of images and landmarks")

    for index, entry in enumerate(ds_train[0]):
        # Comment out if you need to rerun, but putting
        # this here for accidents:
        # break

        train_test_choice = random.randint(0, 100)
        output_folder = data_folder + "train/"
        if train_test_choice > train_percent:
            output_folder = data_folder + "test/"

        #this is the actual code
        num_entries -= 1
        image_array = entry['image'].numpy()
        im = Image.fromarray(image_array)
        im = im.resize(out_size)
        image_filename = output_folder + str(index) + ".jpeg"
        im.save(image_filename)

        facial_features = entry['landmarks_2d']
        facial_features = facial_features * im.size[0]
        facial_features = tf.cast(facial_features, dtype=tf.int32)
        # print("Before")
        # print(facial_features.shape)
        # print(type(facial_features))
        # print(facial_features)
        # output_facial_features = []
        # for x,y in facial_features:
        #     x = int(x * im.size[0])
        #     y = int(y * im.size[1])
        #     output_facial_features.append((x, y))
        # facial_features = tf.convert_to_tensor(output_facial_features, dtype=tf.int32)
        # print("After")
        # print(facial_features.shape)
        # print(type(facial_features))
        # print(facial_features)
        # break
        np_filename = output_folder + str(index) + ".npy"
        np.save(np_filename, facial_features)
        if num_entries == 0:
            break
        elif num_entries % percent == 0:
            print((100 - int(num_entries / percent)), "% complete")

    print("Finished export of images and landmarks")


if __name__ == "__main__":
    import_300lw_data()
