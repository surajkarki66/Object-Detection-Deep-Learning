import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from tensorflow.keras.regularizers import l2

from batch_normalization import BatchNormalization


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


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


def darknet19(input_data=None):
    """ Darknet 19 Architecture """
    if input_data is not None:
        x = convolutional(input_data, (3, 3, 3, 16))
        x = MaxPool2D(2, 2, 'same')(x)
        x = convolutional(x, (3, 3, 16, 32))
        x = MaxPool2D(2, 2, 'same')(x)
        x = convolutional(x, (3, 3, 32, 64))
        x = MaxPool2D(2, 2, 'same')(x)
        x = convolutional(x, (3, 3, 64, 128))
        x = MaxPool2D(2, 2, 'same')(x)
        x = convolutional(x, (3, 3, 128, 256))
        route_1 = x  # (26, 26, 255)
        x = MaxPool2D(2, 2, 'same')(x)
        x = convolutional(x, (3, 3, 256, 512))
        x = MaxPool2D(2, 1, 'same')(x)
        x = convolutional(x, (3, 3, 512, 1024))  # (13, 13, 255)

        return route_1, x

    else:
        return "Please provide input_data"
