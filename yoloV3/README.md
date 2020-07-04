# YOLO
## Introduction

In 2015, Redmon J et al. Proposed the YOLO network, which is characterized by combining the candidate box generation and classification regression into a single step. When predicting, the feature map is divided into 7x7 cells, and each cell is predicted, which greatly reduces the calculation complexity. Accelerate the speed of target detection, frame rate up to 45 fps!

After a lapse of one year, Redmon J once again proposed YOLOv2. Compared with the previous generation, the mAP on the VOC2007 test set increased from 67.4% to 78.6%. However, because a cell is only responsible for predicting a single object, facing the goal of overlap, the recognition was not good enough.

## YOLO V3
Finally, in April 2018, the author released the third version of YOLOv3.The mAP-50 on the COCO dataset was increased from 44.0% of YOLOv2 to 57.9%. Compared with RetinaNet with 61.1% mAP, RetinaNet has an input size of 500. In the case of × 500, the detection speed is about 98 ms/frame, while YOLOv3 has a detection speed of 29 ms/frame when the input size is 416 × 416.

### 1) Idea
The author treats the object detection problem as a regression problem in the YOLO algorithm and divides the image into an S × S grid. If the center of a target falls into a grid, the grid is responsible for detecting the target.

![1](https://user-images.githubusercontent.com/50628520/86329318-b275ba00-bc65-11ea-84f0-ba7a6c913770.jpg)

Each grid will output bounding box, confidence, and class probability map. among them:

 i) The bounding box contains 4 values: x, y, w, h, (x, y) represents the center of the box. (W, h) represents the width and height of the box;

 ii) Confidence indicates the probability of containing objects in this prediction box, which is actually the IoU value between the prediction box and the real box;

 iii) The class probability indicates the class probability of the object, and the YOLOv3 uses a two-class method.

 ### 2) Architecture

As its name suggests, YOLO (You Only Look Once) applies a single forward pass neural network to the whole image and predicts the bounding boxes and their class probabilities as well. This technique makes YOLO quite fast without losing a lot of accuracies.

As mentioned in the original paper , YOLOv3 has 53 convolutional layers called Darknet-53 is shown in the following figure, which is mainly composed of Convolutional and Residual structures. It should be noted that the last three layers Avgpool, Connected and softmax layer are used for classification training on the Imagenet dataset. When we use the Darknet-53 layer to extract features from the picture, these three layers are not used.

[link to YOLOv3 paper!](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

![Darknet-53](https://user-images.githubusercontent.com/50628520/86330362-4dbb5f00-bc67-11ea-8aec-651e2f0b9d48.png)

![yolo_architecture (1)](https://user-images.githubusercontent.com/50628520/86331353-b3f4b180-bc68-11ea-91ce-b58df3ed251d.png)

From the above architecture image, you can see that YOLO makes detection in 3 different scales in order to accommodate different objects size by using strides of 32, 16, and 8. This means, if we’ll feed an input image of size 416 x 416, YOLOv3 will make detection on the scale of 13 x 13, 26 x 26, and 52 x 52.

For the first scale, YOLOv3 downsamples the input image into 13 x 13 and makes a prediction at the 82nd layer. The 1st detection scale yields a 3-D tensor of size 13 x 13 x 255.

After that, YOLOv3 takes the feature map from layer 79 and applies one convolutional layer before upsampling it by a factor of 2 to have a size of 26 x 26. This upsampled feature map is then concatenated with the feature map from layer 61. The concatenated feature map is then subjected to a few more convolutional layers until the 2nd detection scale is performed at layer 94. The second prediction scale produces a 3-D tensor of size 26 x 26 x 255.

The same design is again performed one more time to predict the 3rd scale. The feature map from layer 91 is added one convolutional layer and is then concatenated with a feature map from layer 36. The final prediction layer is done at layer 106 yielding a 3-D tensor of size 52 x 52 x 255. In summary, Yolo predicts over 3 different scales detection, so if we feed an image of size 416x416, it produces 3 different output shape tensor, 13 x 13 x 255, 26 x 26 x 255, and 52 x 52 x 255.

![yolo_structure (1)](https://user-images.githubusercontent.com/50628520/86331781-45fcba00-bc69-11ea-95c3-feb2a9e39e78.png)

### 3) Residual module
The most significant feature of the residual module is the use of a short cut mechanism (somewhat similar to the short circuit mechanism in the circuit) to alleviate the gradient disappearance problem caused by increasing the depth in the neural network, thereby making the neural network easier to optimize. It uses identity mapping to establish a direct correlation channel between input and output so that the network can concentrate on learning the residual between input and output.

<img width="501" alt="residual" src="https://user-images.githubusercontent.com/50628520/86332633-80b32200-bc6a-11ea-9716-8213c7af6511.png">


![prediction](https://user-images.githubusercontent.com/50628520/86333983-58c4be00-bc6c-11ea-8d2c-86ff84f2106d.jpg)

For example, The red ROI in the upper left corner of the original image is mapped by CNN, and only one point is obtained on the feature map space, but this point has 85 channels. So, the dimension of ROI has changed from the original [32, 32, 3] to the current 85-dimension.

This is actually an 85-dimensional feature vector obtained after the CNN network performs feature extraction on the ROI. The first 4 dimensions of this feature vector represent candidate box information, the middle dimension represents the probability of judging the presence or absence of objects, and the following 80 dimensions represent the classification probability information for 80 categories.

### 4) Multi-scale detection
YOLO performs coarse, medium, and fine meshing of the input image to enable the prediction of large, medium, and small objects, respectively. If the size of the input picture is 416X416, then the coarse, medium, and fine grid sizes are 13x13, 26x26, and 52x52 respectively. In this way, it is scaled by 32, 16 and 8 times in length and width respectively:

![multi-scale_pred](https://user-images.githubusercontent.com/50628520/86334721-5151e480-bc6d-11ea-9065-f39292edb979.jpg)

### 5) Dimension of bounding box

The output of the three branches of the YOLOv3 network will be sent to the decode function to decode the channel information of the Feature Map. In the following picture: the black dotted box represents the a priori box (anchor), and the blue box represents the prediction box. The dimensions of the bounding box are predicted by applying a log-space transformation to the output and then multiplying with an anchor:

<img width="560" alt="decode_anchor" src="https://user-images.githubusercontent.com/50628520/86334995-9f66e800-bc6d-11ea-98b5-e8a3733b55bd.png">

 i) b denote the length and width of the prediction frame respectively, and P denote the    length and width of the a priori frame respectively.

 ii) t represents the offset of the center of the object from the upper left corner of the grid, and C represents the coordinates of the upper left corner of the grid.

### 6) NMS processing
Non-Maximum Suppression, as the name implies, suppresses elements that are not maximal. NMS removes those bounding boxes that have a higher overlap rate and a lower score. The algorithm of NMS is straightforward, and the iterative process is as follows:

 i) Process 1: Determine whether the number of bounding boxes is greater than 0, if not, then end the iteration;

 ii) Process 2: Select the bounding box A with the highest score according to the score order and remove it;

 iii) Process 3: Calculate the IoU of this bounding box A and all remaining bounding boxes and remove those bounding boxes whose IoU value is higher than the threshold, repeat the above steps;

 <img width="779" alt="nms_example" src="https://user-images.githubusercontent.com/50628520/86335595-6ed37e00-bc6e-11ea-95a2-4734395f97b9.png">

# yolov3(pretrained weight-> trained on coco dataset)
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# Predicted Output
![output](https://user-images.githubusercontent.com/50628520/86440430-22e60f00-bd2a-11ea-8d3b-afd1d6efa5fd.jpg)

# Predicted vido output
![output](https://user-images.githubusercontent.com/50628520/86506962-00b8c380-bdf4-11ea-8b78-669c85713c9a.gif)
