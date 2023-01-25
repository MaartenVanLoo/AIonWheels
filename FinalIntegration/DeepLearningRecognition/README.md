# Adaptive Cruise Control Camera Object detection using YOLOv5


## 1. Features
Detection of cars, pedestrians and traffic lights using YOLOv5.
## 2. Getting Started
### 2.1. Requirements
To run the classification model, the following packages are required:

gitpython <br>
ipython <br>
matplotlib>=3.2.2 <br>
numpy>=1.18.5 <br>
opencv-python>=4.1.1 <br>
Pillow>=7.1.2 <br>
psutil  # system resources <br>
PyYAML>=5.3.1 <br>
requests>=2.23.0 <br>
scipy>=1.4.1 <br>
thop>=0.1.1 <br>
torch>=1.7.0  <br>
torchvision>=0.8.1 <br>
tqdm>=4.64.0 <br>


### 2.2. Data Collection
The annotated images were generated using the CARLA simulator and processed using Roboflow and can be found through this link: https://universe.roboflow.com/carla-zteyu/carla-car-finder/dataset/16
### 2.3. How to train
Training the model is done using the available scripts on the YOLOv5 Github page.
Example of training for 10 epochs on the collected data:
```
git clone https://github.com/ultralytics/yolov5 
cd yolov5
pip install -r requirements.txt
python train.py --weights yolov5m.pt --data data.yaml --batch 8 --epochs 10
```
In this case the default yolov5m.pt weights are used, however it is also possible to continue training on the checkpoint that we have provided by running the following command:
```
python train.py --weights /FinalIntegration/models/best22_12.pt --data data.yaml --batch 8 --epochs 10
```

## References
[1] YOLOv5 object detection: https://github.com/ultralytics/yolov5 <br>
[2] Roboflow image processing: https://universe.roboflow.com/carla-zteyu/carla-car-finder/dataset/1 <br>

