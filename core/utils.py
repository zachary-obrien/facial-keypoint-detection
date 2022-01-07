from scipy.stats import multivariate_normal
import numpy as np
from matplotlib.pyplot import imshow
import imageio
import imgaug.augmenters as iaa
import cv2
import keras

data_folder = '/Users/zacharyobrien/thesis_work/thesis_dataset/'

# plots literal xy coordinates on the given image
def plot_literal_points(data, points, show=True):
    size = len(data) - 1
    color = (0, 255, 255)
    h = size
    w = len(data[0]) - 1

    for x, y in points:
        # x = int(xr * w)
        # y = int(yr * h)
        x = int(x)
        y = int(y)

        x = max(x, 0)
        x = min(x, w - 1)
        y = max(y, 0)
        y = min(y, h - 1)

        data[y][x] = color

        if y > 0:
            data[y - 1][x] = color
        if x > 0:
            data[y][x - 1] = color
        if x < w:
            data[y][x + 1] = color
        if y < h:
            data[y + 1][x] = color
    if show:
        imshow(data.astype(np.uint8))


# plots the normalized (0 to 1) coordinates on the given image
def plot_keypoints(data, points, show=True):
    size = len(data) - 1
    color = (0, 255, 0)
    h = size
    w = len(data[0]) - 1

    for xr, yr in points:
        x = int(xr * w)
        y = int(yr * h)

        x = max(x, 0)
        x = min(x, w - 1)
        y = max(y, 0)
        y = min(y, h - 1)

        data[y][x] = color

        if y > 0:
            data[y - 1][x] = color
        if x > 0:
            data[y][x - 1] = color
        if x < w:
            data[y][x + 1] = color
        if y < h:
            data[y + 1][x] = color
    if show:
        imshow(data.astype(np.uint8))


# takes a list of heatmaps and returns a list of
# keypoint coordinates
def heatmaps_to_keypoints(hms):
    # max confidences and corresponding coordinates
    max_values = [0 for _ in range(68)]
    max_xy = [[0, 0] for _ in range(68)]

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
        keypoints.append([x / 64, y / 64])

    return keypoints


# the model outputs more than one set of heatmaps
# this function can be used to average those into one set
def average_heatmaps(hms):
    return np.mean(hms, axis=0)

    # these next four functions are for creating the gaussian heatmaps from
    # the batch of coordinate labels
    # code source: https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html
def xy_to_heatmap(downscaling, xy):
    mean = [int(xy[1] / downscaling), int(xy[0] / downscaling)]
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
        image = image[:, crop_amt:-crop_amt]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # bgr_to_rgb(image)
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
                kps = heatmaps_to_keypoints(preds[-1, i])
                plot_keypoints(frame, kps, show=False)

            imageio.mimsave(data_folder + 'data/awing_40k_epoch{}.gif'.format(epoch), frames)


# reduces the loss every n epochs
def scheduler(epoch, lr):
    n = 11

    if epoch > 1 and (epoch - 1) % n == 0:
        return lr - 5e-5
    else:
        return lr

