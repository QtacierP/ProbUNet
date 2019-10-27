import cv2
import matplotlib.pyplot as plt
import h5py
import glob
from scipy.io import loadmat, savemat
import imageio
from tqdm import tqdm
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.misc import imsave
from skimage.measure import label, regionprops
import pandas as pd
from utils import *
from data.common import AbstractDataLoader

class DataLoader(AbstractDataLoader):
    def __init__(self, args):
        super(DataLoader, self).__init__(args)


    def get_train(self):
        imgs_train = load_hdf5(self.hdf5_dir + '/train/train.hdf5')
        gt_train = load_hdf5(self.hdf5_dir + '/train/train_gt.hdf5')
        return imgs_train, gt_train


    def get_val(self):
        imgs_val = load_hdf5(self.hdf5_dir + '/test/test.hdf5')
        gt_val = load_hdf5(self.hdf5_dir + '/test/test_gt.hdf5')
        return imgs_val[:25], gt_val[:25]


    def get_test(self):
        imgs_test = load_hdf5(self.hdf5_dir + '/test/test.hdf5')
        gt_test = load_hdf5(self.hdf5_dir + '/test/test_gt.hdf5')
        return imgs_test[25:], gt_test[25:]


















