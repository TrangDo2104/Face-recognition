# Face-recognition

## Dataset

The dataset for custom training can be downloaded from the following link:

[Download Dataset](https://vis-www.cs.umass.edu/lfw/)

Please download the dataset in 'All images as gzipped tar file' and extract it to the root directory of this project.

## Object Detection Model

The object detection model used in this project is YOLOv3-SPP. The pre-trained weights can be downloaded from the following link:

[Download YOLOv3-SPP Weights](https://drive.google.com/file/d/1h2g_wQ270_pckpRCHJb9K78uDf-2PsPd/view?usp=sharing)

Please download the weights and place them in the "detector/yolo/data/" folder.

## Pose Tracking Model

For pose tracking, we use an object tracking model. The pre-trained weights for the JDE-1088x608-uncertainty model can be downloaded from the following link:

[Download JDE-1088x608-uncertainty Weights](https://drive.google.com/file/d/1oek1aj9t7pTi1u70nSlwx0qNVWvEvRrf/view?usp=sharing)

Please download the weights and place them in the "detector/tracker/data/" folder.

## Pretrained Model

We also use a pretrained Fast.res50.pt model. The model can be downloaded from the following link:

[Download Fast.res50.pt](https://drive.google.com/file/d/1oek1aj9t7pTi1u70nSlwx0qNVWvEvRrf/view?usp=sharing)

Please download the model and place it in the root directory of this project.

## Usage

To use the object detection and pose tracking models, run the following command:
