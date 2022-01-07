import numpy as np
from PIL import Image
import keras
import json
from scipy.stats import multivariate_normal
import tensorflow as tf

"""
    generator that loads and augments batches of images for training
    imgaug handles augmenting the images and their corresponding keypoints
"""

data_folder = '/Users/zacharyobrien/thesis_work/thesis_dataset/'
#from tensorflow.keras.utils
class train_generator(keras.utils.all_utils.Sequence):
    def __init__(self):
        # the parameters are saved and loaded from a json because tf
        # requires the generator to be instantiated with zero arguments
        # in order to convert it to a tf.dataset
        with open(data_folder + 'train_params.json') as f:
            params = json.load(f)
        self.ids = params['ids']
        self.batch_size = params['batch_size']
        self.image_shape = params['image_shape']
        self.downscaling = 256 / 64

        # image augmentation pipeline
        # self.aug = iaa.Sequential([
        #     iaa.CropToAspectRatio(1),
        #     iaa.Resize({"height": 256, "width": 256}),
        #     iaa.Affine(scale=(0.7, 1.3), rotate=(-40, 40))])

        # constants used to create the gaussian heatmaps for the labels
        self.x = np.arange(0, 64, 1, float)  ## (width,)
        self.y = np.arange(0, 64, 1, float)[:, np.newaxis]  ## (height,1)

    def __len__(self):
        return (np.ceil(len(self.ids) / float(self.batch_size))).astype(np.int)

    # for some reason tf was saying that the generators were running out before
    # they were supposed to when using interleave so I just made this function loop infinitely
    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        while 1:
            for item in (self[i] for i in range(len(self))):
                yield item

    # loads the image and the corresponding labels
    def load(self, num):
        img = np.asarray(Image.open(num + '.jpeg'))
        return (img, np.load(num + '.npy'))

    # these next four functions are for creating the gaussian heatmaps from
    # the batch of coordinate labels
    # code source: https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html
    def xy_to_heatmap(self, xy):
        mean = [int(xy[1] / self.downscaling), int(xy[0] / self.downscaling)]
        # print('{} -> {}'.format([xy[1], xy[0]], mean))

        pos = np.dstack(np.mgrid[0:64:1, 0:64:1])
        rv = multivariate_normal(mean=mean, cov=4)

        return rv.pdf(pos)

    def gaussian_k(self, x0, y0, sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        return np.exp(-((self.x - x0) ** 2 + (self.y - y0) ** 2) / (2 * sigma ** 2))

    def generate_hm(self, height, width, landmarks, s=3):
        """ Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
        """

        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks), dtype=np.float32)
        for i in range(Nlandmarks):
            if not np.array_equal(landmarks[i], [-1, -1]):

                hm[:, :, i] = self.gaussian_k(landmarks[i][0],
                                              landmarks[i][1],
                                              s, height, width)
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm

    def get_y_as_heatmap(self, kps, height, width, sigma):
        y_train = []
        for i in range(kps.shape[0]):
            y_train.append(self.generate_hm(height, width, kps[i], sigma))

        y_train = np.array(y_train)

        return y_train

    # given a batch index loads the batch of images and keypoints,
    # augments them, creates the heatmaps and returns the batch of
    # images and heatmaps
    def __getitem__(self, idx):
        batch_ids = self.ids[idx * self.batch_size: (idx + 1) * self.batch_size]
        files = list(map(self.load, batch_ids))
        batch_x = [tup[0] for tup in files]
        batch_y = [tup[1] for tup in files]

        # batch_x, batch_y = self.aug(images=batch_x, keypoints=batch_y)

        batch_y = np.array(batch_y)
        batch_y = batch_y / self.downscaling
        batch_y = self.get_y_as_heatmap(batch_y, 64, 64, 2)

        return np.asarray(batch_x), batch_y


# Same as the train generator, but loads different parameters
# and does less augmentation (just crops and resizes to fit model)
class val_generator(keras.utils.all_utils.Sequence):
    def __init__(self):
        with open(data_folder + 'val_params.json') as f:
            params = json.load(f)
        self.ids = params['ids']
        self.batch_size = params['batch_size']
        self.image_shape = params['image_shape']
        self.downscaling = 256 / 64
        # self.aug = iaa.Sequential([
        #     iaa.CropToAspectRatio(1),
        #     iaa.Resize({"height": 256, "width": 256})])

        self.x = np.arange(0, 64, 1, float)  ## (width,)
        self.y = np.arange(0, 64, 1, float)[:, np.newaxis]  ## (height,1)

    def __len__(self):
        return (np.ceil(len(self.ids) / float(self.batch_size))).astype(np.int)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        while 1:
            for item in (self[i] for i in range(len(self))):
                yield item

    def xy_to_heatmap(self, xy):
        mean = [int(xy[1] / self.downscaling), int(xy[0] / self.downscaling)]
        # print('{} -> {}'.format([xy[1], xy[0]], mean))

        pos = np.dstack(np.mgrid[0:64:1, 0:64:1])
        rv = multivariate_normal(mean=mean, cov=4)

        return rv.pdf(pos)

    def load(self, num):
        # img = Image.open(num + '.png')
        # if type(img.getpixel((0, 0))) == int:
        #     rgbimg = Image.new("RGB", img.size)
        #     rgbimg.paste(img)
        #     img = rgbimg

        img = np.asarray(Image.open(num + '.jpeg'))
        return (img, np.load(num + '.npy'))
        # img = np.asarray(img)
        # img = np.asarray(Image.open(num + '.jpeg'))
        # return (img, np.load(num + '.npy'))

    def gaussian_k(self, x0, y0, sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        return np.exp(-((self.x - x0) ** 2 + (self.y - y0) ** 2) / (2 * sigma ** 2))

    def generate_hm(self, height, width, landmarks, s=3):
        """ Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
        """

        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks), dtype=np.float32)
        for i in range(Nlandmarks):
            if not np.array_equal(landmarks[i], [-1, -1]):

                hm[:, :, i] = self.gaussian_k(landmarks[i][0],
                                              landmarks[i][1],
                                              s, height, width)
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm

    def get_y_as_heatmap(self, kps, height, width, sigma):
        y_train = []
        for i in range(kps.shape[0]):
            y_train.append(self.generate_hm(height, width, kps[i], sigma))

        y_train = np.array(y_train)

        return y_train

    def __getitem__(self, idx):
        batch_ids = self.ids[idx * self.batch_size: (idx + 1) * self.batch_size]
        files = list(map(self.load, batch_ids))
        batch_x = [tup[0] for tup in files]
        batch_y = [tup[1] for tup in files]

        # batch_x, batch_y = self.aug(images=batch_x, keypoints=batch_y)

        batch_y = np.array(batch_y)
        batch_y = batch_y / self.downscaling
        batch_y = self.get_y_as_heatmap(batch_y, 64, 64, 2)
        return np.asarray(batch_x), batch_y

# loader functions for the generators needed by tensorflow
# in order to use interleave
def get_train_dataset(self):
    self = tf.data.Dataset.from_generator(
        train_generator,
        output_types = (tf.float32, tf.float32))
    return self

def get_val_dataset(self):
    self = tf.data.Dataset.from_generator(
        val_generator,
        output_types = (tf.float32, tf.float32))
    return self
