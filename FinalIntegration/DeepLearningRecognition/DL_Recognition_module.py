import multiprocessing
import os
import queue
import time
from typing import Optional, Callable, Any, Tuple, Mapping

import cv2
import torch

from .models.common import DetectMultiBackend
from .utils.plots import Annotator, Colors
from .utils.general import non_max_suppression
import numpy as np


class DeepLearningRecognition(object):
    def __init__(self, carlaWorld, config=None) -> None:
        if not config:
            config = dict()
        super().__init__()
        self.config = config
        self._carlaWorld = carlaWorld
        self._agent = carlaWorld.getPlayer()

        self.device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        self._model_name = "/"
        self._model = self._load_model(config.get("model_path",""), config.get("hubconf_path","")).to(self.device)
        self._model.eval()
        self.names = self._model.names


        self.detections = None
        self.detected_image = None

        #yolo settings
        self.conf_thres = self.config.get("conf_threshold")
        self.iou_thres = self.config.get("iou_threshold")
        self.max_detect = self.config.get("max_detections")

        #drawing boxes:
        self.line_thickness = 3
        self.colors = Colors()
        self.hide_labels = self.config.get("hide_labels",False)
        self.hide_confidence = self.config.get("hide_confidence",False)

        #adjust speed
        self.current_max_speed = None
        self.is_orange_light = False
        self.is_red_light = False

    def detect(self):
        start = time.time()
        # Load image from sensor
        sensor = self._carlaWorld.get_sensor("Camera")

        if sensor is None:
            print(f"Inference time object detection:{(time.time() - start)*1000:4.0f} ms")
            return
        image = sensor.getState()
        if image is None:
            print(f"Inference time object detection:{(time.time() - start)*1000:4.0f} ms")
            return
        image = image.copy()
        image = cv2.resize(image, (640,640))

        tensor = torch.tensor(image).to(self.device)
        tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2).float()
        tensor /= 255
        y = self._model.forward(tensor)

        if isinstance(y, (list, tuple)):
            pred = self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            pred = self.from_numpy(y)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_detect)

        detected_objects = []
        annotator = Annotator(image.copy(),line_width=self.line_thickness, example=str(self.names))
        for i, det in enumerate(pred):
            #print results
            s = ""
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print(s)

            #draw results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls) #class
                label = None if self.hide_labels else \
                    (self.names[c] if self.hide_confidence else f'{self.names[c]}'f' {conf:.2f}')
                detected_objects.append(self.names[c])
                annotator.box_label(xyxy, label, color=self.colors(c, True))

            #check for signs and lights:
        self.speedLimit(detected_objects)
        self.trafficLights(detected_objects)

        self.detections = detected_objects
        self.detected_image = annotator.result()
        print(f"Inference time object detection:{(time.time() - start)*1000:4.0f} ms")

    def _load_model(self, filename, hubconf_path):
        if not (os.path.exists(filename) and os.path.isfile(filename)):
            print("Model file not found")
            return None
        if not (os.path.exists(hubconf_path + "/hubconf.py")):
            print("Hubconf not found")
            return None
        #model = torch.hub.load(hubconf_path, 'custom', path=filename, source='local')
        #from .models.experimental import attempt_load
        #model = attempt_load(filename)
        _, self._model_name = os.path.split(filename)
        model = DetectMultiBackend(filename, device=self.device)
        return model

    #name is het name of the detected object
    def speedLimit(self,detections):
        if "traffic_sign_30" in detections:
            print("Detected:traffic_sign_30")
            self.current_max_speed = 30
        elif "traffic_sign_60" in detections:
            print("Detected:traffic_sign_60")
            self.current_max_speed = 60
        elif "traffic_sign_90" in detections:
            print("Detected:traffic_sign_90")
            self.current_max_speed = 90

    def trafficLights(self,detections):
        if "traffic_light_yellow" in detections:
            print("Detected:orange_light")
            self.is_orange_light = True
        else:
            self.is_orange_light = False

        if "traffic_light_red" in detections:
            print("Detected:red_light")
            self.is_red_light = True
        else:
            self.is_red_light = False


    def getModelName(self) -> str:
        return self._model_name

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x





