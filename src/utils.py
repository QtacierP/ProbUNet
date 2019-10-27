# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize
import scipy
from skimage.measure import label, regionprops






def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["image"][()]

def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset('image', data=arr, dtype=arr.dtype)


def dataset_normalized(imgs):
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=0) # turn into 4D arrays
    #assert (imgs.shape[3] == 3)  # check the channel is 3
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)

    print('std', imgs_std)
    imgs_mean = np.mean(imgs)
    print('mean', imgs_mean)
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = (imgs[i] - imgs_mean) / imgs_std
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))
    return imgs_normalized, imgs_mean, imgs_std

def inference_normalized(imgs, imgs_mean=51.4170, imgs_std=64.7685):
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=0) # turn into 4D arrays
    #assert (imgs.shape[3] == 3)  # check the channel is 3
    imgs_normalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = (imgs[i] - imgs_mean) / imgs_std
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))
    return imgs_normalized



