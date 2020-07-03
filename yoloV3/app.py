import tensorflow as tf

from detection import Detection
from utils import load_yolo_weights
from model.yolov3 import complete_yolov3
from configs import *

if __name__ == "__main__":
    input_size = YOLO_INPUT_SIZE
    Darknet_weights = YOLO_DARKNET_WEIGHTS

    # input paths
    image_path = "test_images/1.jpg"
    video_path = "test_video/test.mp4"

    # outputs path
    image_output_path = 'output.jpg'
    video_output_path = 'output.mp4'
    live_output_path = 'live_output.mp4'

    yolo = complete_yolov3(input_size)
    load_yolo_weights(yolo, Darknet_weights)  # use Darknet weights
    detection = Detection(yolo, YOLO_COCO_CLASSES)

    # Image detection
    detection.detect_image(image_path, output_path=image_output_path, input_size=input_size,
                          show=True, rectangle_colors=(8, 8, 9))

    # Video Detection
    #detection.detect_video(video_path, output_path = video_output_path,
    #                       input_size=input_size, show=True, rectangle_colors=(8, 8, 9))

    # Live Detection
    #detection.live_detection(live_output_path, input_size=input_size, show=True, rectangle_colors=(8, 8, 9))
