import numpy as np
import tensorflow as tf

from configs import *
from utils import read_class_names
from model.darknet19 import darknet19, convolutional, upsample

STRIDES         = np.array(YOLO_STRIDES)
ANCHORS         = (np.array(YOLO_ANCHORS).T/STRIDES).T


def tiny_YOLOV3(input_layer, NUM_CLASS):
    # After the input layer enters the Darknet-19 network, we get two branches
    route_1, conv = darknet19(input_layer)

    conv = convolutional(conv, (1, 1, 1024, 256))
    conv_lobj_branch = convolutional(conv, (3, 3, 256, 512))

    # conv_lbbox is used to predict large-sized objects , Shape = [None, 26, 26, 255]
    conv_lbbox = convolutional(
        conv_lobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    # upsampling process does not need to learn, thereby reducing the network parameter
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)
    conv_mobj_branch = convolutional(conv, (3, 3, 128, 256))
    # conv_mbbox is used to predict medium size objects, shape = [None, 13, 13, 255]
    conv_mbbox = convolutional(
        conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]


def complete_tiny_YOLOV3(input_size=416, channels=3, training=False, CLASSES=YOLO_COCO_CLASSES):
    NUM_CLASS = len(read_class_names(CLASSES))
    input_layer = tf.keras.layers.Input([input_size, input_size, channels])

    conv_tensors = tiny_YOLOV3(input_layer, NUM_CLASS)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training:
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)
   # print(output_tensors[1].shape)
    tiny_YoloV3 = tf.keras.Model(input_layer, output_tensors)
    return tiny_YoloV3


def decode(conv_output, NUM_CLASS, i=0):
    # where i = 0, 1 or 2 to correspond to the three grid scales
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(
        conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # offset of center position
    # Prediction box length and width offset
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    # confidence of the prediction box
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    # category probability of the prediction box
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size, dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [
                      batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    # object box calculates the predicted confidence
    pred_conf = tf.sigmoid(conv_raw_conf)
    # calculating the predicted probability category box object
    pred_prob = tf.sigmoid(conv_raw_prob)

    # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
