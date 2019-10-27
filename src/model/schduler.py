# -*- coding: utf-8 -*-
'''
2019/8/29 : add multi GPU support and F1 callback
2019/8/30 : add AUC/kappa callback
'''
import keras.backend as K
from keras.callbacks import  Callback
import math
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, cohen_kappa_score
import warnings
from math import pow, floor
import numpy as np
from utils import *
from keras.utils import to_categorical
from option import args


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

def step_decay(epoch):
    x = 5e-3
    if epoch >= 50: x /= 5.0
    if epoch >= 80: x /= 5.0
    return x

def exponent_decay(epoch):
    init_lrate = 1e-4
    drop = 1e-6
    epochs_drop = 10
    lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
    return lrate

def training_exponent_decay(init_lrate, epoch):
    drop = 1e-6
    epochs_drop = 10
    lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
    print('Exponent Decay Lr to', lrate)
    return lrate


def custom_schedule(epochs):
    if epochs <= 5:
        lr = 1e-4
    elif epochs <= 10:
        lr = 5e-4
    elif epochs <= 20:
        lr = 2.5e-4
    else:
        lr = 1e-5
    return lr

def training_custom_schedule(epochs):
    if epochs <= 5:
        lr = 1e-3
    elif epochs <= 10:
        lr = 5e-4
    elif epochs <= 20:
        lr = 2.5e-4
    else:
        lr = 1e-3
    return lr

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)


def boolMap(arr):
    if arr > 0.5:
        return 1
    else:
        return 0

def custom_schedule_v2(epochs):
        if epochs <= 5:
            lr = 1e-3
        elif epochs <= 10:
            lr = 5e-4
        elif epochs <= 20:
            lr = 2.5e-4
        elif epochs <= 30:
            lr = 5e-5
        else:
            lr = 1e-5
        return lr

class MyCheckPoint(ModelCheckpoint):
    def __init__(self, model,  filepath, opt = 'f1', monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, train_data=None,  train_label=None,
                 mode='auto', period=1, multil_gpus=False, val_data=None, val_label=None):

        self.file_path = filepath
        self.mutil_gpus = multil_gpus
        self.single_model = model
        self.opt = opt

        super(MyCheckPoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)
        self.validation_data = []
        self.validation_data.append(val_data)
        self.validation_data.append(val_label)

        self.train_data = []
        self.train_data.append(train_data)
        self.train_data.append(train_label)


    def set_model(self, model):
        if self.mutil_gpus:
            self.model = self.single_model
        else:

            self.model = model

    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.best_one = 0
        self.val_recalls = []
        self.val_precisions = []
        self.val_aucs = []
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs=None):

        val_score = self.model.predict(self.validation_data[0])
        val_predict = (val_score >= 0.5)
        average = 'binary'
        if not args.violence:
            average = 'macro'
            val_predict = to_categorical(val_predict)
        val_targ = self.validation_data[1].flatten()
        _val_f1 = f1_score(val_targ, val_predict, average=average)
        _val_recall = recall_score(val_targ, val_predict, average=average)
        _val_precision = precision_score(val_targ, val_predict, average=average)
        _val_auc = roc_auc_score(val_targ, val_score, average=average)
        _val_kappa = cohen_kappa_score(val_targ, val_predict)

        train_score = self.model.predict(self.train_data[0]).flatten()
        train_predict = (train_score >= 0.5)
        train_targ = self.train_data[1].flatten()
        _train_f1 = f1_score(train_targ, train_predict)
        _train_recall = recall_score(train_targ, train_predict)
        _train_precision = precision_score(train_targ, train_predict)
        _train_auc = roc_auc_score(train_targ, train_score)
        _train_kappa = cohen_kappa_score(train_targ, train_predict)


        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_aucs.append(_val_auc)
        self.val_kappas.append(_val_kappa)
        if self.opt == 'f1':
            one = _val_f1
        elif self.opt == 'auc':
            one = _val_auc
        else:
            one = _val_kappa
        print('[Train]: F1 Score : {}, Precision : {}, Recall : {},  AUC : {}, Kappa : {}'.
              format(_train_f1, _train_precision, _train_recall, _train_auc, _train_kappa))
        print('[Val]: F1 Score : {}, Precision : {}, Recall : {},  AUC : {}, Kappa : {}'.
              format(_val_f1, _val_precision, _val_recall, _val_auc, _val_kappa))
        if one > self.best_one:
            self.model.save_weights(self.file_path, overwrite=True)
            print('val_{} improved from {} to {}'.format(self.opt, self.best_one,one))
            self.best_one = one
        else:
            print("val {}: {}, but did not improve from the best {} {}".
                  format(self.opt, one, self.opt, self.best_one))
        return



def get_check_point(filepath, model, model_name,
                    verbose=1,
                    mode='auto',
                    save_best_only=True):
    if 'GAN' in model_name:
        monitor = 'val_categorical_accuracy'
    else:
        monitor = 'val_loss'

    return ParallelModelCheckpoint(
        filepath=filepath,
        model=model,
        verbose=verbose,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only)
