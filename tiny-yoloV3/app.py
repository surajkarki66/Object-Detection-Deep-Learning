import tensorflow as tf

from detection import Detection
from utils import load_yolo_weights
from model.tiny_yoloV3 import complete_tiny_YOLOV3
from configs import *

if __name__ == "__main__":
    input_size = YOLO_INPUT_SIZE
    Darknet_weights = YOLO_DARKNET_TINY_WEIGHTS 

    # input paths
    image_path = "test/images/1.jpg"
    video_path = "test/videos/test.mp4"

    # outputs path
    image_output_path = 'output.jpg'
    video_output_path = 'output.mp4'
    live_output_path = 'live_output.mp4'

    tiny_yoloV3 = complete_tiny_YOLOV3(input_size)
    load_yolo_weights(tiny_yoloV3, Darknet_weights)  # use Darknet weights
    detection = Detection(tiny_yoloV3, YOLO_COCO_CLASSES)

    # Image detection
    #detection.detect_image(image_path, output_path=image_output_path, input_size=input_size,
     #                     show=True, rectangle_colors=(8, 8, 9))

    # Video Detection
    detection.detect_video(video_path, output_path = video_output_path,
                           input_size=input_size, show=True, rectangle_colors=(8, 8, 9))

    # Live Detection
    #detection.live_detection(live_output_path, input_size=input_size, show=True, rectangle_colors=(8, 8, 9))
