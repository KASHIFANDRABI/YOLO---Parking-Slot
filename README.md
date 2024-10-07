# YOLO---Parking-Slot
This project uses YOLOv5 for parking space detection, trained on a custom dataset to identify available parking spots in real-time from images or video feeds. It aims to improve parking management systems by providing accurate and fast detection of vehicles and open spaces

A list of commonly used resources that I find helpful are listed in the acknowledgements.

Build:
This section should list any major frameworks that you built your project.

  1)YOLOv5.
  2)pytorch.
  3)RoboFlow Annotate (Data Preparation & Labelling).



Prerequisites:
These libraries are needed for to run this project locally.

Core Libraries:
1) PyTorch (torch) - Used for building and training the YOLOv5 model.
2) YOLOv5 - The main repository and framework used for object detection. It is built on top of PyTorch.
3) OpenCV (cv2) - Used for image and video processing, including reading frames, drawing bounding boxes, and displaying results.
4) NumPy (numpy) - Provides support for numerical operations and array manipulations, commonly used for handling image data.
5) Matplotlib - Used for visualizing training progress and performance metrics like loss and accuracy.
6) Glob - It helps in getting a list of files in a directory that match the given pattern.

Procedure
  1) Get Dataset from Roboflow - Use this link to get dataset for the model. [Click Here](https://universe.roboflow.com/muhammad-syihab-bdynf/parking-space-ipm1b/dataset/4)
  2) Open the YOLO_V05_model_Parking.ipynb Notebook to setup yaml file and train YOLO model.
  3) Monitor Training: You can visualize the training process and monitor the metrics like loss, mAP, precision, and recall.
  4) Saving Model: Once training is complete, the results of the model will be saved in runs/train/yolov5s_results.
