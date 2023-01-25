import multiprocessing
import os
import threading

import queue

import time
from typing import Optional, Callable, Any, Tuple, Mapping

import cv2
import torch

from .models.common import DetectMultiBackend
from .utils.plots import Annotator, Colors
from .utils.general import non_max_suppression
import numpy as np

import logging


class DistributedRecognition(object):
    def __init__(self, carlaWorld, config=None) -> None:
        if not config:
            config = dict()
        super().__init__()
        self.config = config
        self._carlaWorld = carlaWorld
        self._agent = carlaWorld.getPlayer()

        self.device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self._model_name = "/"
        # self._model = self._load_model(config.get("model_path",""), config.get("hubconf_path","")).to(self.device)
        # self._model.eval()

        # internal states
        self.current_frame = -1
        self.detections = None
        self.detected_image = None

        # yolo settings
        self.conf_thres = self.config.get("conf_threshold")
        self.iou_thres = self.config.get("iou_threshold")
        self.max_detect = self.config.get("max_detections")

        # drawing boxes:
        self.line_thickness = 3
        self.colors = Colors()
        self.hide_labels = self.config.get("hide_labels", False)
        self.hide_confidence = self.config.get("hide_confidence", False)

        self._worker_queue = queue.Queue()

        self._logger = logging.getLogger()

        self.sensor = self._carlaWorld.get_sensor("Camera")
        from FinalIntegration.Utils.Sensor import AsyncCamera
        if not isinstance(self.sensor, AsyncCamera):
            raise TypeError("Sensor must be an async Camera")

        # launching workers
        self._workers = []
        self._workers.append(RecognitionWorker(0, self.sensor, self))
        # self._workers.append(RecognitionWorker(1, self.sensor, self))
        for worker in self._workers:
            self._model_name = worker.getModelName()
            worker.start()

        # adjust speed
        self.current_max_speed = None
        self.is_orange_light = False
        self.is_red_light = False

    def detect(self):
        start = time.time()
        while not self._worker_queue.empty():
            frame, det, det_im, max_speed, red_light, orange_light = self._worker_queue.get()
            if frame <= self.current_frame:
                continue  # older sample, no need to store, can happen because of asynchronous processing
            # adjust speed
            self.current_max_speed = self.current_max_speed if max_speed is None else max_speed
            self.is_orange_light = self.is_orange_light if orange_light is None else orange_light
            self.is_red_light = self.is_red_light if red_light is None else red_light
            self.current_frame = frame
            self.detections = det
            self.detected_image = det_im
        self._logger.info(f"Object detection {self.current_frame} :\t{(time.time() - start) * 1000:4.0f} ms")

    def getModelName(self) -> str:
        return self._model_name

    def cleanup(self):
        for worker in self._workers:
            worker.stop()
            worker.join()


class RecognitionWorker(threading.Thread):
    def __init__(self, id, sensor, parent: DistributedRecognition):
        self._logger = logging.getLogger()
        self._id = id
        self._logger.debug(f"Loading Recognition worker - {id}")
        super().__init__()
        self.input_queue = sensor.getQueue()

        self._output_queue = parent._worker_queue
        self.device = parent.device

        # yolo settings
        self.conf_thres = parent.conf_thres
        self.iou_thres = parent.iou_thres
        self.max_detect = parent.max_detect

        # drawing boxes:
        self.line_thickness = 3
        self.colors = Colors()
        self.hide_labels = parent.hide_labels
        self.hide_confidence = parent.hide_confidence

        # create model after all variables have been set
        self.model_path = parent.config.get("model_path", "")
        self.hubconf_path = parent.config.get("hubconf_path", "")

        _, self._model_name = os.path.split(self.model_path)

        # adjust speed
        self.current_max_speed = None
        self.is_orange_light = None
        self.is_red_light = None
        self.min_speed_confidence = 0.9

        self.running = True

    def stop(self):
        self.running = False

    def run(self) -> None:
        self._model = self._load_model(self.model_path, self.hubconf_path)
        self._model.eval()
        self.names = self._model.names
        while self.running:
            try:
                frame, image = self.input_queue.get(block=True, timeout=60)
                print(f"Recognition get image: {frame}")
                if not self.input_queue.empty():
                    print(f"skipping image: {frame}")
                    continue
            except Exception:
                continue

            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                detections, detected_image = self.detect(image)
            self._output_queue.put(
                (frame, detections, detected_image, self.current_max_speed, self.is_red_light, self.is_orange_light))

            #Todo: comment this line
            #threading.Thread(target = self.saveImage, args = (frame, image,detected_image)).start()

            print(f"Done with image: {frame}")

    def _load_model(self, filename, hubconf_path):
        if not (os.path.exists(filename) and os.path.isfile(filename)):
            print("Model file not found")
            return None
        if not (os.path.exists(hubconf_path + "/hubconf.py")):
            print("Hubconf not found")
            return None
        # model = torch.hub.load(hubconf_path, 'custom', path=filename, source='local')
        # from .models.experimental import attempt_load
        # model = attempt_load(filename)
        _, self._model_name = os.path.split(filename)
        model = DetectMultiBackend(filename, device=self.device)
        return model

    def detect(self, image):
        image = cv2.resize(image, (640, 640))

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
        annotator = Annotator(image.copy(), line_width=self.line_thickness, example=str(self.names))
        for i, det in enumerate(pred):
            # print results
            s = ""
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print(s)

            # draw results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # class
                label = None if self.hide_labels else \
                    (self.names[c] if self.hide_confidence else f'{self.names[c]}'f' {conf:.2f}')
                detected_objects.append((self.names[c], conf))
                annotator.box_label(xyxy, label, color=self.colors(c, True))

        self.speedLimit(detected_objects)
        self.trafficLights(detected_objects)

        detections = detected_objects
        detected_image = annotator.result()



        return detections, detected_image

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def speedLimit(self, detections):
        self.current_max_speed = None  # when not changed, this will be the final value
        for detection in detections:
            label = detection[0]
            conf = detection[1]
            if conf < self.min_speed_confidence:  # must be very certain before adjusting speed
                continue
            if "traffic_sign_30" in label:
                print("Detected:traffic_sign_30")
                self.current_max_speed = 30 / 3.6
            elif "traffic_sign_60" in label:
                print("Detected:traffic_sign_60")
                self.current_max_speed = 60 / 3.6
            elif "traffic_sign_90" in label:
                print("Detected:traffic_sign_90")
                self.current_max_speed = 90 / 3.6

    def trafficLights(self, detections):
        labels = []
        for detection in detections:
            labels.append(detection[0])
            conf = detection[1]
        if "traffic_light_orange" in labels:
            print("Detected:orange_light")
            self.is_orange_light = True
        else:
            self.is_orange_light = False

        if "traffic_light_red" in labels:
            print("Detected:red_light")
            self.is_red_light = True
        else:
            self.is_red_light = False

    def getModelName(self) -> str:
        return self._model_name

    def saveImage(self, frame, image, detected_image):
        filename = f"{frame}_"
        cv2.imwrite(filename + "original.png", image)
        cv2.imwrite(filename + "detected.png", detected_image)


