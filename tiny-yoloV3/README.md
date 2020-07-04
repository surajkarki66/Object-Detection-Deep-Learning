# YOLO-Tiny
## Introduction
To help make YOLOv3 even faster, Redmon et al. (the creators of YOLO), defined a variation of the YOLO architecture called YOLOv3-Tiny.

### 1) Idea
The author treats the object detection problem as a regression problem in the YOLO algorithm and divides the image into an S × S grid. If the center of a target falls into a grid, the grid is responsible for detecting the target.


 ### 2) Architecture
 YOLOv3‐Tiny instead of Darknet53 has a backbone of the Darknet19, the structure of it is shown in the following image:

![darknet19](https://user-images.githubusercontent.com/50628520/86506187-dadbf080-bdec-11ea-89e1-10a316cf1547.png)

![tiny-yolo](https://user-images.githubusercontent.com/50628520/86506207-fd6e0980-bdec-11ea-930b-323bec62ded5.png)



# Predicted Output
Accuracy is not good as compared to YOLOV3.

![output](https://user-images.githubusercontent.com/50628520/86506223-27bfc700-bded-11ea-8e25-b141705b7a49.jpg)

# Predicted Video Output

![tiny](https://user-images.githubusercontent.com/50628520/86506865-285b5c00-bdf3-11ea-8688-248ff37d6370.gif)
