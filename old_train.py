import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from hourglass import StackedHourglassNetwork
# from binary_hourglass import binary_stacked_hourglass
import keras
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.batches import UnnormalizedBatch
#import torchfile
from multiprocessing import Pool
from PIL import Image
from scipy.stats import multivariate_normal
import json
import os
from collections import defaultdict
from matplotlib.pyplot import imshow
from numpy import log
from tensorflow import pow, abs, cumsum, reduce_sum, greater, reduce_mean
import cv2
import random


"""
    Contains data augmentation pipeline and model training.

    Sorry this script is fairly monolithic because I just dumped the cells
    needed for training models.
"""

data_folder = '/Users/zacharyobrien/thesis_work/thesis_dataset/'
# loads the paths from to the training and validation data
def load_filenames():
    # return list(range(len(os.listdir('./data/preprocessed_training/')) // 2))
    nums = defaultdict(bool)

    for f in os.listdir(data_folder + 'train/'):
        num = f[:-4]
        nums[num] = True
        
    train = [data_folder + 'train/' + f for f in list(nums.keys())]
    
    nums = defaultdict(bool)

    for f in os.listdir(data_folder + 'test/'):
        #num = re.search('(.+)\..{2,3}', f)[1]
        num = f[:-4]
        nums[num] = True
    
    
    return train, [data_folder + 'test/' + f for f in list(nums.keys())]

# loss function that helps learning with heatmaps
# the issue with using MSE on the heatmaps is that most of the values are
# zero so once the model is outputting all zeros the gradients get very small.
# Adaptive wing loss weights the non-zero values of the ground truth heatmaps
# much higher.
# theory: https://arxiv.org/pdf/1904.07399v3.pdf
# code source: https://github.com/andrewhou1/Adaptive-Wing-Loss-for-Face-Alignment/blob/master/hourglasstensorflow/hourglass_tiny.py
def adaptive_wing_loss(labels, output):
    alpha = 2.1
    omega = 14
    epsilon = 1
    theta = 0.5
    with tf.name_scope('adaptive_wing_loss'):
        x = output - labels
        theta_over_epsilon_tensor = tf.fill(tf.shape(labels), theta/epsilon)
        A = omega*(1/(1+pow(theta_over_epsilon_tensor, alpha-labels)))*(alpha-labels)*pow(theta_over_epsilon_tensor, alpha-labels-1)*(1/epsilon)
        C = theta*A-omega*log(1+pow(theta_over_epsilon_tensor, alpha-labels))
        absolute_x = abs(x)
        losses = tf.where(greater(theta, absolute_x), omega*log(1+pow(absolute_x/epsilon, alpha-labels)), A*absolute_x-C)
        loss = reduce_mean(reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss

# plots literal xy coordinates on the given image
def plot_literal_points(data, points, show=True):
    size = len(data) - 1
    color = (0,255,255)
    h = size
    w = len(data[0]) - 1
    
    for x, y in points:
        #x = int(xr * w)
        #y = int(yr * h)
        x = int(x)
        y = int(y)
        
        x = max(x, 0)
        x = min(x, w - 1)
        y = max(y, 0)
        y = min(y, h-1)
        
        data[y][x] = color

        if y > 0:
            data[y-1][x] = color
        if x > 0:
            data[y][x-1] = color
        if x < w:
            data[y][x+1] = color
        if y < h:
            data[y+1][x] = color
    if show:
        imshow(data.astype(np.uint8))

# plots the normalized (0 to 1) coordinates on the given image
def plot_keypoints(data, points, show=True):
    size = len(data) - 1
    color = (0,255,0)
    h = size
    w = len(data[0]) - 1
    
    for xr, yr in points:
        x = int(xr * w)
        y = int(yr * h)
        
        x = max(x, 0)
        x = min(x, w - 1)
        y = max(y, 0)
        y = min(y, h-1)
        
        data[y][x] = color

        if y > 0:
            data[y-1][x] = color
        if x > 0:
            data[y][x-1] = color
        if x < w:
            data[y][x+1] = color
        if y < h:
            data[y+1][x] = color
    if show:
        imshow(data.astype(np.uint8))

"""
    generator that loads and augments batches of images for training
    imgaug handles augmenting the images and their corresponding keypoints
"""
class train_generator(keras.utils.Sequence):
    def __init__(self):
        # the parameters are saved and loaded from a json because tf
        # requires the generator to be instantiated with zero arguments
        # in order to convert it to a tf.dataset
        with open(data_folder + 'train_params.json') as f:
            params = json.load(f)
        self.ids = params['ids']
        self.batch_size = params['batch_size']
        self.image_shape = params['image_shape']
        self.downscaling = 256/64

        # image augmentation pipeline
        self.aug = iaa.Sequential([
            iaa.CropToAspectRatio(1),
            iaa.Resize({"height": 256, "width": 256}),
            iaa.Affine(scale=(0.7, 1.3), rotate=(-40, 40))])
        
        # constants used to create the gaussian heatmaps for the labels
        self.x = np.arange(0, 64, 1, float) ## (width,)
        self.y = np.arange(0, 64, 1, float)[:, np.newaxis] ## (height,1)
        
    def __len__(self) :
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
        img = np.asarray(Image.open(num + '.jpg'))
        return (img, np.load(num + '.npy'))

    # these next four functions are for creating the gaussian heatmaps from
    # the batch of coordinate labels
    # code source: https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html
    def xy_to_heatmap(self, xy):
        mean = [int(xy[1] / self.downscaling), int(xy[0] / self.downscaling)]
        #print('{} -> {}'.format([xy[1], xy[0]], mean))

        pos = np.dstack(np.mgrid[0:64:1, 0:64:1])
        rv = multivariate_normal(mean=mean, cov=4)

        return rv.pdf(pos)
    
    def gaussian_k(self, x0,y0,sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        return np.exp(-((self.x-x0)**2 + (self.y-y0)**2) / (2*sigma**2))

    def generate_hm(self, height, width ,landmarks,s=3):
        """ Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
        """

        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
        for i in range(Nlandmarks):
            if not np.array_equal(landmarks[i], [-1,-1]):
            
                hm[:,:,i] = self.gaussian_k(landmarks[i][0],
                                        landmarks[i][1],
                                        s,height, width)
            else:
                hm[:,:,i] = np.zeros((height,width))
        return hm

    def get_y_as_heatmap(self, kps,height,width, sigma):
        y_train = []
        for i in range(kps.shape[0]): 
            y_train.append(self.generate_hm(height, width, kps[i], sigma))

        y_train = np.array(y_train)
    
    
        return y_train

    # given a batch index loads the batch of images and keypoints, 
    # augments them, creates the heatmaps and returns the batch of
    # images and heatmaps
    def __getitem__(self,idx):
        batch_ids = self.ids[idx * self.batch_size : (idx+1) * self.batch_size]
        files = list(map(self.load, batch_ids))
        batch_x = [tup[0] for tup in files]
        batch_y = [tup[1] for tup in files]

        batch_x, batch_y = self.aug(images=batch_x, keypoints=batch_y)
        
        batch_y = np.array(batch_y)
        batch_y = batch_y / self.downscaling
        batch_y = self.get_y_as_heatmap(batch_y, 64, 64, 2)
        
        return np.asarray(batch_x), batch_y

# Same as the train generator, but loads different parameters
# and does less augmentation (just crops and resizes to fit model)
class val_generator(keras.utils.Sequence):
    def __init__(self):
        with open(data_folder + 'val_params.json') as f:
            params = json.load(f)
        self.ids = params['ids']
        self.batch_size = params['batch_size']
        self.image_shape = params['image_shape']
        self.downscaling = 256/64
        self.aug = iaa.Sequential([
            iaa.CropToAspectRatio(1),
            iaa.Resize({"height": 256, "width": 256})])
        
        self.x = np.arange(0, 64, 1, float) ## (width,)
        self.y = np.arange(0, 64, 1, float)[:, np.newaxis] ## (height,1)
        

    def __len__(self) :
        return (np.ceil(len(self.ids) / float(self.batch_size))).astype(np.int)
    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        while 1:
            for item in (self[i] for i in range(len(self))):
                yield item
    def xy_to_heatmap(self, xy):
        mean = [int(xy[1] / self.downscaling), int(xy[0] / self.downscaling)]
        #print('{} -> {}'.format([xy[1], xy[0]], mean))

        pos = np.dstack(np.mgrid[0:64:1, 0:64:1])
        rv = multivariate_normal(mean=mean, cov=4)

        return rv.pdf(pos)
    
    def load(self, num):
        img = Image.open(num + '.png')
        if type(img.getpixel((0,0))) == int:
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg
        img = np.asarray(img)
        #img = np.asarray(Image.open(num + '.jpg'))
        return (img, np.load(num + '.npy'))


    def gaussian_k(self, x0,y0,sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        return np.exp(-((self.x-x0)**2 + (self.y-y0)**2) / (2*sigma**2))

    def generate_hm(self, height, width ,landmarks,s=3):
        """ Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
        """

        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
        for i in range(Nlandmarks):
            if not np.array_equal(landmarks[i], [-1,-1]):
            
                hm[:,:,i] = self.gaussian_k(landmarks[i][0],
                                        landmarks[i][1],
                                        s,height, width)
            else:
                hm[:,:,i] = np.zeros((height,width))
        return hm

    def get_y_as_heatmap(self, kps,height,width, sigma):
        y_train = []
        for i in range(kps.shape[0]): 
            y_train.append(self.generate_hm(height, width, kps[i], sigma))

        y_train = np.array(y_train)
    
    
        return y_train

    def __getitem__(self,idx):
        batch_ids = self.ids[idx * self.batch_size : (idx+1) * self.batch_size]
        files = list(map(self.load, batch_ids))
        batch_x = [tup[0] for tup in files]
        batch_y = [tup[1] for tup in files]

        batch_x, batch_y = self.aug(images=batch_x, keypoints=batch_y)
        
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

# takes a list of heatmaps and returns a list of 
# keypoint coordinates
def heatmaps_to_keypoints(hms):
    # max confidences and corresponding coordinates
    max_values = [0 for _ in range(68)]
    max_xy = [[0,0] for _ in range(68)]

    # find the coordinats of the pixel with the highest
    # value (confidence) from each heatmap
    for y in range(64):
        for x in range(64):
            for idx in range(68):
                if hms[y][x][idx] > max_values[idx]:
                    max_values[idx] = hms[y][x][idx]
                    max_xy[idx] = [x, y]

    # normalize the keypoints from 0 to 1
    keypoints = []
    for i, xy in enumerate(max_xy):
        x, y = xy
        keypoints.append([x/64, y/64])
    
    return keypoints

# the model outputs more than one set of heatmaps
# this function can be used to average those into one set
def average_heatmaps(hms):
    return np.mean(hms, axis=0)

# returns a list of frames from a video prepared to be
# passed to the model
# takes the path to a video, extracts each frame,
# crops each frame to 1:1 and resizes them to 256x256
def extract_frames(filename):
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    
    h, w, _ = image.shape
    crop_amt = (w - h) // 2  
    aug = iaa.Resize({"height": 256, "width": 256})
    
    frames = []

    while success:
        image = image[:,crop_amt:-crop_amt]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #bgr_to_rgb(image)
        frames.append(image)
        success, image = vidcap.read()

    return aug(images=frames), frames

# custom keras callback to create a test gif every n epochs
# the gifs are created from the frames of a video with the keypoints 
# predicted by the model plotted on them
class gifCallback(keras.callbacks.Callback):
    def __init__(self):
        frames, _ = extract_frames(data_folder + 'test_video.mp4')
        self.frames = frames
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        n = 3

        if (epoch + 1) % n == 0:
            frames = np.array(self.frames)
            preds = np.array(self.model.predict(frames))

            for i, frame in enumerate(frames):
                kps = heatmaps_to_keypoints(preds[-1,i])
                plot_keypoints(frame, kps, show=False)

            imageio.mimsave('./data/awing_40k_epoch{}.gif'.format(epoch), frames)

# reduces the loss every n epochs
def scheduler(epoch, lr):
    n = 11

    if epoch > 1 and (epoch - 1) % n == 0:
        return lr - 5e-5
    else:
        return lr

# the callbacks used during training
# lr scheduler, model checkpoint to save the weights after
# each record low validation loss, and the custom gifCallback
callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),
            tf.keras.callbacks.ModelCheckpoint(
                data_folder + 'models/weights.40k_{epoch:02d}-{val_loss:.2f}.hdf5',
                monitor="val_loss",
                save_weights_only=True,
                verbose=1,
                save_best_only=True,
                mode="auto",
                save_freq="epoch",
            ), gifCallback()]

# using RMSprop optimizer
optimizer = tf.keras.optimizers.RMSprop(2.5e-4)

# create the stacked hourglass model with the given number
# of stacked hourglass modules
# code source: https://github.com/ethanyanjiali/deep-vision/tree/master/Hourglass/tensorflow
model = StackedHourglassNetwork(num_stack=2)
model.compile(optimizer=optimizer, loss=adaptive_wing_loss)

# from each of the generators create a pair of interleaved datasets
# tensorflow can automatically multiprocess interleaved datasets
# so that while batches can be loaded and processed ahead of time
interleaved_train = tf.data.Dataset.range(2).interleave(
                    get_train_dataset,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
interleaved_val = tf.data.Dataset.range(2).interleave(
                    get_val_dataset,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                )

# load training and validation filenames
train_ids, val_ids = load_filenames()
random.shuffle(train_ids)
random.shuffle(val_ids)

# store generator parameters
batch_size = 8
image_shape = (256,256,3)

train_params = {
    'ids': train_ids,
    'batch_size': batch_size,
    'image_shape': image_shape
}
val_params = {
    'ids': val_ids,
    'batch_size': batch_size,
    'image_shape': image_shape
}
json.dump(train_params, open(data_folder + 'train_params.json', 'w'))
json.dump(val_params, open(data_folder + 'val_params.json', 'w'))

# train the model
model.fit(interleaved_train,
                   steps_per_epoch = int(len(train_ids) // batch_size),
                   epochs = 50,
                    verbose=1,
                    validation_data=interleaved_val,
                    validation_steps=int(len(val_ids) // batch_size),
                   callbacks=callbacks)

#model.save_weights('./data/models/10k_weights_55_epochs_noflip')