from keras.layers import *
from keras.models import Model
import tensorflow as tf

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              dilated_rate=(1, 1),
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name, dilation_rate=dilated_rate)(x)
    if not use_bias:
        bn_axis = 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        if activation == 'LeakyReLU':
            x = LeakyReLU(alpha=0.2, name=ac_name)(x)
        else:
            x = Activation(activation, name=ac_name)(x)
    return x
def UNet_backbone(inputs, num_class):
    # 288 x 288 x 32
    conv1 = conv2d_bn(inputs, kernel_size=3, filters=32, name='conv1_1')
    conv1 = conv2d_bn(conv1, kernel_size=3, filters=32, name='conv1_2')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 144 x 144 x 64
    conv2 = conv2d_bn(pool1, kernel_size=3, filters=64, name='conv2_1')
    conv2 = conv2d_bn(conv2, kernel_size=3, filters=64, name='conv2_2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 72 x 72 x 128
    conv3 = conv2d_bn(pool2, kernel_size=3, filters=128, name='conv3_1')
    conv3 = conv2d_bn(conv3, kernel_size=3, filters=128, name='conv3_2')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 36 x 36 x 256
    conv4 = conv2d_bn(pool3, kernel_size=3, filters=256, name='conv4_1')
    conv4 = conv2d_bn(conv4, kernel_size=3, filters=256, name='conv4_2')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # 36 x 36 x 512
    conv5 = conv2d_bn(pool4, kernel_size=3, filters=512, name='conv5_1')
    conv5 = conv2d_bn(conv5, kernel_size=3, filters=512, name='conv5_2')

    # Up-sampling: 72 x 72 x 256
    up1 = Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(conv5)
    up1 = concatenate([conv4, up1])
    conv6 = conv2d_bn(up1, kernel_size=3, filters=256, name='conv6_1')
    conv6 = conv2d_bn(conv6, kernel_size=3, filters=256, name='conv6_2')

    # Up-sampling: 144 x 144 x 128
    up2 = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(conv6)
    up2 = concatenate([conv3, up2])
    conv7 = conv2d_bn(up2, kernel_size=3, filters=128, name='conv7_1')
    conv7 = conv2d_bn(conv7, kernel_size=3, filters=128, name='conv7_2')

    # Up-sampling: 288 x 288 x 64
    up3 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv7)
    up3 = concatenate([conv2, up3])
    conv8 = conv2d_bn(up3, kernel_size=3, filters=64, name='conv8_1')
    conv8 = conv2d_bn(conv8, kernel_size=3, filters=64, name='conv8_2')

    # Up-sampling: 576 x 576 x 32
    up4 = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(conv8)
    up4 = concatenate([conv1, up4])
    conv9 = conv2d_bn(up4, kernel_size=3, filters=32, name='conv9_1')
    conv9 = conv2d_bn(conv9, kernel_size=3, filters=32, name='conv9_2')

    # Full Connection: 576 x 576 x 3
    conv9 = Conv2D(num_class, (1, 1), activation='relu', padding='same', name='conv9_sotmax')(conv9)
    if num_class == 1:
        output = Activation('sigmoid', name='output')(conv9)
    else:
        output = Activation('softmax', name='output')(conv9)
    return output

def UNetBN_backbone(inputs, num_class, name='', logits=False):
    # 288 x 288 x 32
    conv1 = conv2d_bn(inputs, kernel_size=3, filters=32, name=name+'conv1_1')
    conv1 = conv2d_bn(conv1, kernel_size=3, filters=32, name=name+'conv1_2')
    pool1 = conv2d_bn(conv1, filters=32, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name+'pool1')

    # 144 x 144 x 64
    conv2 = conv2d_bn(pool1, kernel_size=3, filters=64, name=name+'conv2_1')
    conv2 = conv2d_bn(conv2, kernel_size=3, filters=64, name=name+'conv2_2')
    pool2 = conv2d_bn(conv2, filters=64, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name+'pool2')

    # 72 x 72 x 128
    conv3 = conv2d_bn(pool2, kernel_size=3, filters=128, name=name+'conv3_1')
    conv3 = conv2d_bn(conv3, kernel_size=3, filters=128, name=name+'conv3_2')
    pool3 = conv2d_bn(conv3, filters=128, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name+'pool3')

    # 36 x 36 x 256
    conv4 = conv2d_bn(pool3, kernel_size=3, filters=256, name=name+'conv4_1')
    conv4 = conv2d_bn(conv4, kernel_size=3, filters=256, name=name+'conv4_2')
    pool4 = conv2d_bn(conv4, filters=256, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name+'pool4')

    # 36 x 36 x 512
    conv5 = conv2d_bn(pool4, kernel_size=3, filters=512, name=name+'conv5_1')
    conv5 = conv2d_bn(conv5, kernel_size=3, filters=512, name=name+'conv5_2')

    # Up-sampling: 72 x 72 x 256
    up1 = Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same', name=name+'up1')(conv5)
    up1 = concatenate([conv4, up1], name=name+'up1_c')
    conv6 = conv2d_bn(up1, kernel_size=3, filters=256, name=name+'conv6_1')
    conv6 = conv2d_bn(conv6, kernel_size=3, filters=256, name=name+'conv6_2')

    # Up-sampling: 144 x 144 x 128
    up2 = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same', name=name+'up2')(conv6)
    up2 = concatenate([conv3, up2], name=name+'up2_c')
    conv7 = conv2d_bn(up2, kernel_size=3, filters=128, name=name+'conv7_1')
    conv7 = conv2d_bn(conv7, kernel_size=3, filters=128, name=name+'conv7_2')

    # Up-sampling: 288 x 288 x 64
    up3 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same', name=name+'up3')(conv7)
    up3 = concatenate([conv2, up3], name=name+'up3_c')
    conv8 = conv2d_bn(up3, kernel_size=3, filters=64, name=name+'conv8_1')
    conv8 = conv2d_bn(conv8, kernel_size=3, filters=64, name=name+'conv8_2')

    # Up-sampling: 576 x 576 x 32
    up4 = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same', name=name+'up4')(conv8)
    up4 = concatenate([conv1, up4], name=name+'up4_c')
    conv9 = conv2d_bn(up4, kernel_size=3, filters=32, name=name+'conv9_1')
    conv9 = conv2d_bn(conv9, kernel_size=3, filters=32, name=name+'conv9_2')
    # Full Connection: 576 x 576 x 3
    output = Conv2D(num_class, (1, 1), activation='relu', padding='same', name=name+'conv9_sotmax')(conv9)
    if not logits:
        if num_class == 1:
            output = Activation('sigmoid', name=name+'output')(conv9)
        else:
            output = Activation('softmax', name=name+'output')(conv9)
    return output

def encoder(inputs, name=''):
    '''
    :param inputs:
    :return: Encoding features
    '''
    conv1 = conv2d_bn(inputs, kernel_size=3, filters=32, name=name + 'conv1_1')
    conv1 = conv2d_bn(conv1, kernel_size=3, filters=32, name=name + 'conv1_2')
    pool1 = conv2d_bn(conv1, filters=32, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name + 'pool1')


    conv2 = conv2d_bn(pool1, kernel_size=3, filters=64, name=name + 'conv2_1')
    conv2 = conv2d_bn(conv2, kernel_size=3, filters=64, name=name + 'conv2_2')
    pool2 = conv2d_bn(conv2, filters=64, strides=2, kernel_size=4,
                      activation='LeakyReLU', name= name + 'pool2')


    conv3 = conv2d_bn(pool2, kernel_size=3, filters=128, name=name + 'conv3_1')
    conv3 = conv2d_bn(conv3, kernel_size=3, filters=128, name=name + 'conv3_2')
    pool3 = conv2d_bn(conv3, filters=128, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name + 'pool3')


    conv4 = conv2d_bn(pool3, kernel_size=3, filters=256, name=name + 'conv4_1')
    conv4 = conv2d_bn(conv4, kernel_size=3, filters=256, name=name + 'conv4_2')
    pool4 = conv2d_bn(conv4, filters=256, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name + 'pool4')


    conv5 = conv2d_bn(pool4, kernel_size=3, filters=512, name=name + 'conv5_1')
    conv5 = conv2d_bn(conv5, kernel_size=3, filters=512, name=name + 'conv5_2')

    return conv5