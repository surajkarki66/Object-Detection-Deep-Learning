import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from tensorflow.keras.regularizers import l2

from batch_normalization import BatchNormalization
from configs import *

STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T/STRIDES).T


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    """ Typical conv -> batchnorm -> LeakyRelu """
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        conv = LeakyReLU(alpha=0.1)(conv)

    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    """ Residual identity block """
    shortcut = input_layer
    conv = convolutional(input_layer, filters_shape=(
        1, 1, input_channel, filter_num1))
    conv = convolutional(conv, filters_shape=(
        3, 3, filter_num1,   filter_num2))

    residual_output = shortcut + conv
    return residual_output


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


def darknet53(input_data):
    """ Darknet53 architecture """
    x = convolutional(input_data, (3, 3,  3,  32))
    x = convolutional(x, (3, 3, 32,  64), downsample=True)

    for _ in range(1):
        x = residual_block(x,  64,  32, 64)

    x = convolutional(x, (3, 3,  64, 128), downsample=True)

    for _ in range(2):
        x = residual_block(x, 128,  64, 128)

    x = convolutional(x, (3, 3, 128, 256), downsample=True)

    for _ in range(8):
        x = residual_block(x, 256, 128, 256)

    route_1 = x
    x = convolutional(x, (3, 3, 256, 512), downsample=True)

    for _ in range(8):
        x = residual_block(x, 512, 256, 512)

    route_2 = x
    x = convolutional(x, (3, 3, 512, 1024), downsample=True)

    for _ in range(4):
        x = residual_block(x, 1024, 512, 1024)

    return route_1, route_2, x
