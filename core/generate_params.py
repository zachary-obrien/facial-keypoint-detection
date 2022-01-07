import random
from collections import defaultdict
import os
import json
from params.project_config import data_folder


def load_filenames():
    # return list(range(len(os.listdir('./data/preprocessed_training/')) // 2))
    nums = defaultdict(bool)

    train_filenames = os.listdir(data_folder + 'train/')
    for f in train_filenames:
        if f.endswith(".jpeg"):
            num = f[:-5]
            nums[num] = True

    train = []
    for g in list(nums.keys()):
        train.append(data_folder + 'train/' + g)
        # if g == '3363':
        #     print("found 3363")


    nums2 = defaultdict(bool)

    test_filenames = os.listdir(data_folder + 'test/')
    for f in test_filenames:
        # num = re.search('(.+)\..{2,3}', f)[1]
        if f.endswith(".jpeg"):
            num2 = f[:-5]
            nums2[num2] = True

    test = []
    for g in list(nums2.keys()):
        test.append(data_folder + 'test/' + g)
        # if g == '3363':
        #     print("found 3363")


    return train, test


def generate_param_files():
    # load training and validation filenames
    train_ids, val_ids = load_filenames()
    # random.shuffle(train_ids)
    # random.shuffle(val_ids)

    # store generator parameters
    batch_size = 2
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

if __name__ == "__main__":
    generate_param_files()