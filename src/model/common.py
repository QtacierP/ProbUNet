# -*- coding: utf-8 -*-
from keras.layers import *
from keras_preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow_probability as tfp
tfd = tfp.distributions
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, \
    LearningRateScheduler
from importlib import import_module
from model.backbone import *
from model.schduler import get_check_point

def get_model(args):
    print(import_module('model.' + args.model.lower()).MyModel(args))
    return import_module('model.' + args.model.lower()).MyModel(args)

class Network():
    def __init__(self, args, name=None):
        self.class_num = args.class_num
        self.c = args.n_colors
        self.h = args.height
        self.w = args.width
        self.args = args
        self.n_gpus = args.n_gpus
        if name != None:
            self.model_name = name
        else:
            self.model_name = args.model
        self.model_path = args.model_path
        self.is_train = not args.test
        self.build_model()
        self.save()
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):

        self.callbacks.append(
            get_check_point(
                filepath=self.args.model_path + self.args.model + '_best_weights.h5',
                model_name = self.model_name,
                model=self.model,
                verbose=1,
                mode='auto',
                save_best_only=True))

        self.callbacks.append(
            TensorBoard(
                log_dir=self.args.model_path + 'log/',
                write_images=True,
                write_graph=True,
            )
        )
        self.callbacks.append(
            EarlyStopping(
                patience=self.args.early_stopping
            )
        )


        def custom_schedule(epochs):
            if epochs <= 5:
                lr = 1e-3
            elif epochs <= 10:
                lr = 5e-4
            elif epochs <= 20:
                lr = 2.5e-4
            else:
                lr = 1e-3

            return lr

        self.callbacks.append(
            LearningRateScheduler(
                custom_schedule
            )
        )
    def save(self):
        """
        Save the checkpoint, with the path defined in configuration
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Saving model...")
        json_string = self.model.to_json()
        open(self.model_path + self.model_name + '_architecture.json', 'w').write(json_string)
        print("[INFO] Model saved")

    def load(self):
        """
        Load the checkpoint, with the path defined in configuration
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Loading model checkpoint ...\n")
        self.model.load_weights(self.model_path + self.model_name + '_best_weights.h5')
        print("[INFO] Model loaded")

    def build_model(self):
        pass





def AxisAlignedConvGaussian(inputs, name, latent_dim=6, channel_num=32, act='relu'):

    encoding = encoder(inputs, name=name+'encoding')
    encoding = Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keep_dims=True), name=name+'mean_encoding')(encoding)

    mu_log_sigma = conv2d_bn(encoding, filters= 2 * latent_dim,
                             kernel_size=1, activation=act, name=name+'m_l')

    mu_log_sigma_squeeze = Lambda(lambda x: tf.squeeze(x, axis=[1, 2])
                                  , name=name+'m_l_s')(mu_log_sigma)
    mu = Lambda(lambda x: x[:, : latent_dim], name=name+'mu')(mu_log_sigma_squeeze)
    log_sigma = Lambda(lambda x: x[:, latent_dim: ], name=name+'log_sigma')(mu_log_sigma_squeeze)

    return [mu, log_sigma]




def keras_tile(x):
    features = x[0]
    z = x[1]
    print(features)
    shp = features.get_shape()
    print(shp)
    multiples = [1, shp[1], shp[2], 1]

    print(multiples)
    if len(z.get_shape()) == 2:
        z = tf.expand_dims(z, axis=1)
        z = tf.expand_dims(z, axis=1)
    # broadcast latent vector to spatial dimensions of the image/feature tensor
    broadcast_z = tf.tile(z, multiples)

    print('broad is ',  broadcast_z)
    print(broadcast_z)
    features = tf.concat([features, broadcast_z], axis=-1)
    return features

def FcombDecoder(features, z, class_num, channel_num=32, conv_num=3, act='relu', name=''):
    features = Lambda(keras_tile, name=name+'features')([features, z]) # expand
    for i in range(conv_num):
        features = conv2d_bn(features, kernel_size=1, filters=channel_num
                             ,activation=act, name=name+'conv_' + str(i))
    logits = conv2d_bn(features, kernel_size=1, filters=class_num, name=name+'logits')
    return logits

def trainGenerator(x, y, aug_dict, batch_size, seed=1):
    # Normal Generator
    input_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    input_generator = input_datagen.flow(x, batch_size=batch_size, seed=seed)
    label_generator = label_datagen.flow(y, batch_size=batch_size, seed=seed)
    train_generator = zip(input_generator, label_generator)
    for (input, label) in train_generator:
        yield (input, label)

def ProbGenerator(x, y, aug_dict, batch_size, seed=1):
    # used for ProbUNet
    input_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    input_generator = input_datagen.flow(x, batch_size=batch_size, seed=seed)
    label_generator = label_datagen.flow(y, batch_size=batch_size, seed=seed)
    train_generator = zip(input_generator, label_generator)
    for (input, label) in train_generator:
        yield ([input, label], label)