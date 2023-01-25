# Adaptive Cruise Control Camera Object detection using YOLOv5


## 1. Features

## 2. Getting Started
### 2.1. Requirements

### 2.2. Data Collection

### 2.3. How to train
Training the model is done using the available scripts on the YOLOv5 Github page.
Example of training for 10 epochs on collected data:

```
git clone https://github.com/ultralytics/yolov5 
cd yolov5
pip install -r requirements.txt
python train.py --weights yolov5m.pt --data data.yaml --batch 8 --epochs 10
```
In this case the default yolov5m.pt weights are used, however it is also possible to continue training on the checkpoint that we have provided.
```
python train.py --weights /FinalIntegration/models/best22_12.pt --data data.yaml --batch 8 --epochs 10
```

## References
https://github.com/ultralytics/yolov5
