import time
import cv2
import numpy as np
import tensorflow as tf

from utils import image_preprocess, postprocess_boxes, nms, draw_bbox


class Detection:
    def __init__(self, tiny_YoloV3, CLASSES):
        self.tiny_YoloV3 = tiny_YoloV3
        self.CLASSES = CLASSES

    def detect_image(self, image_path=None, output_path=None, input_size=416, show=False, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
        if image_path is not None:

            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = image_preprocess(np.copy(original_image), [
                input_size, input_size])
            image_data = tf.expand_dims(image_data, 0)

            # it gives output in three different scale
            pred_bbox = self.tiny_YoloV3.predict(image_data)
            print(pred_bbox[0].shape)
            print(pred_bbox[1].shape)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1]))
                         for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            # print(pred_bbox)

            bboxes = postprocess_boxes(
                pred_bbox, original_image, input_size, score_threshold)
            print(bboxes.shape)
            bboxes = nms(bboxes, iou_threshold, method='nms')
            print(bboxes[0].shape)
            print(len(bboxes))

            image = draw_bbox(original_image, bboxes, CLASSES=self.CLASSES,
                              rectangle_colors=rectangle_colors)

            # print(image.shape)
            if output_path is not None:
                cv2.imwrite(output_path, image)
            if show:
                # Show the image
                cv2.imshow("predicted image", image)
                # Load and hold the image
                cv2.waitKey(0)
                # To close the window after the required kill value was provided
                cv2.destroyAllWindows()

            return image

    def detect_video(self, video_path, output_path=None, input_size=416, show=False, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
        times = []
        vid = cv2.VideoCapture(video_path)

        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        # output_path must be .mp4
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))

        while True:
            _, img = vid.read()

            try:
                original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                original_image = cv2.cvtColor(
                    original_image, cv2.COLOR_BGR2RGB)
            except:
                break
            image_data = image_preprocess(np.copy(original_image), [
                                          input_size, input_size])
            image_data = tf.expand_dims(image_data, 0)

            t1 = time.time()
            pred_bbox = self.tiny_YoloV3.predict(image_data)
            t2 = time.time()

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1]))
                         for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(
                pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')

            times.append(t2-t1)
            times = times[-20:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms

            print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

            image = draw_bbox(
                original_image, bboxes, CLASSES=self.CLASSES, rectangle_colors=rectangle_colors)
            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            if output_path is not None:
                out.write(image)
            if show:
                cv2.imshow('output', image)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

        cv2.destroyAllWindows()

    def live_detection(self, output_path=None, input_size=416, show=False, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
        times = []
        vid = cv2.VideoCapture(0)

        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        # output_path must be .mp4
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))

        while True:
            _, frame = vid.read()

            try:
                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_frame = cv2.cvtColor(
                    original_frame, cv2.COLOR_BGR2RGB)
            except:
                break
            image_frame = image_preprocess(np.copy(original_frame), [
                                           input_size, input_size])
            image_frame = tf.expand_dims(image_frame, 0)

            t1 = time.time()
            pred_bbox = self.tiny_YoloV3.predict(image_frame)
            t2 = time.time()

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1]))
                         for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(
                pred_bbox, original_frame, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')

            times.append(t2-t1)
            times = times[-20:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms

            print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

            frame = draw_bbox(
                original_frame, bboxes, CLASSES=self.CLASSES, rectangle_colors=rectangle_colors)
            image = cv2.putText(frame, "Time: {:.1f}FPS".format(fps), (0, 30),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            if output_path is not None:
                out.write(frame)
            if show:
                cv2.imshow('output', frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

        cv2.destroyAllWindows()
