import logging
import os
import queue
import threading
import time
import traceback

import carla
import cv2

from .data_process.kitti_bev_utils import makeBEVMap, get_filtered_lidar, drawRotatedBox
from .utils.evaluation_utils import decode,post_processing
import torch

import FinalIntegration.DeepLearningLidar.config.kitti_config as cnf
import numpy as np

from .models.fpn_resnet import get_pose_net


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)

class DistributedLidar(object):
    def __init__(self, carlaWorld, config = None) -> None:
        if not config:
            config = dict()
        super().__init__()
        self.config = config
        self.debug = config.get("debug",False)
        self.config.down_ratio = 4
        self.config.peak_thresh = 0.2

        self._carlaWorld= carlaWorld
        self._agent = carlaWorld.getPlayer()

        self.device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

        self._model_name = "/"
        self.bev_image = None
        self.current_frame = -1

        self.distance = 110
        self.detected_boxes = None

        self._worker_queue = queue.Queue()
        self._logger = logging.getLogger()

        self.sensor = self._carlaWorld.get_sensor("Lidar")
        from FinalIntegration.Utils.Sensor import AsyncLidar
        if not isinstance(self.sensor, AsyncLidar):
            raise TypeError("Sensor must be an async Lidar")
        # launching workers
        self._workers = []
        self._workers.append(LidarWorker(0, self.sensor, self))
        #self._workers.append(RecognitionWorker(1, self.sensor, self))
        for worker in self._workers:
            worker.start()

    def detect(self):
        start = time.time()
        while not self._worker_queue.empty():
            frame, bev_im, detected_box = self._worker_queue.get()
            if frame <= self.current_frame:
                continue  # older sample, no need to store, can happen because of asynchronous processing
            self.current_frame = frame
            self.bev_image = bev_im
            self.detected_boxes = detected_box
        self._logger.info(f"Lidar detection {self.current_frame} :{(time.time() - start) * 1000:4.0f} ms")

    def getModelName(self)->str:
        return self._model_name
    def setModelName(self, name :str)->None:
        self._model_name = name







class LidarWorker(threading.Thread):
    def __init__(self, id, sensor, parent: DistributedLidar):
        self._logger = logging.getLogger()
        self._id = id
        self._logger.debug(f"Loading Lidar worker - {id}")
        super().__init__()
        self.input_queue = sensor.getQueue()

        self._output_queue = parent._worker_queue
        self.device = parent.device
        self.config = parent.config

        self._model_name = "/"
        self._model = self.create_model(self.config).to(self.device)
        self._model.eval() #set model in evaluation mode
        parent.setModelName(self._model_name)

        self.num_classes = self.config.get('num_classes')
        self.colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
                       [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]

        self._carlaWorld = parent._carlaWorld
        self.debug = parent.debug


    def run(self) -> None:
        while True:
            try:
                frame, image = self.input_queue.get(block=True, timeout=60)
                print(f"Lidar get frame: {frame}")
                if not self.input_queue.empty():
                    print(f"Lidar skipping frame: {frame}")
                    continue
            except Exception:
                traceback.print_exc()
                continue

            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                bev_image, detected_boxes = self.detect(image)
            self._output_queue.put((frame, bev_image, detected_boxes))
            print(f"Lidar done with frame: {frame}")

    def create_model(self,config):
        """Create model based on architecture name"""
        heads = {
            'hm_cen': config.get('num_classes'),
            'cen_offset': config.get('num_center_offset'),
            'direction': config.get('num_direction'),
            'z_coor': config.get('num_z'),
            'dim': config.get('num_dim')
        }

        print('using ResNet architecture with feature pyramid')

        model = get_pose_net(num_layers=config.get('num_layers'), heads=heads, head_conv=config.get('head_conv'),
                             imagenet_pretrained=config.get('imagenet_pretrained'), config=config)
        _, self._model_name = os.path.split(config.get('model_path'))
        return model

    def detect(self, frame):
        start = time.time()

        frame = get_filtered_lidar(frame, cnf.boundary)
        bev_map = makeBEVMap(frame, cnf, cnf.boundary)
        bev_image = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
        with torch.no_grad():
            torch_map = torch.from_numpy(bev_map).to(self.device)
            torch_map = torch_map.unsqueeze(0).float()
            #Todo: forward model
            outputs = self._model(torch_map)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            #Todo: decode result
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=self.config.get('K'))
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, self.config.num_classes, self.config.down_ratio, self.config.peak_thresh)
            detections = detections[0]

            bev_image = self._draw_output_map(bev_image, detections)
            detected_boxes = self._create_bounding_boxes(detections)

        return bev_image, detected_boxes

    def _draw_output_map(self,bev_map, detections):
        # Draw prediction in the image
        pred_map = bev_map
        pred_map = cv2.resize(pred_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        pred_map = self._draw_predictions(pred_map, detections.copy(), self.num_classes)

        # flip bev_map to allign with the image
        pred_map = cv2.flip(pred_map, 1)
        pred_map = cv2.flip(pred_map, 0)
        return pred_map

    def _draw_predictions(self,img, detections, num_classes=3):
        for j in range(num_classes):
            if len(detections[j]) > 0:
                for det in detections[j]:
                    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    drawRotatedBox(img, _x, _y, _w, _l, _yaw, self.colors[int(j)])

        return img

    def _create_bounding_boxes(self, detections):
        #get lidar sensor:
        lidar = self._carlaWorld.get_sensor('Lidar')
        if lidar is None:
            return None #no sensor = no bounding boxes
        lidar_transform = lidar.sensor.get_transform()

        boxes = []

        #add player also to detections for debug reasons:

        for j in range(self.num_classes):

            for det in detections[j]:
                _score, _y, _x, _z, _h, _w, _l, _yaw = det # TODO: swap x and y seems to work, WHYYYYY
                #from pixel coordinates to global coordinates
                #Todo: find this convertion !, check if below is correct
                x_range = (cnf.boundary['minX'],cnf.boundary['maxX'])
                y_range = (cnf.boundary['minY'],cnf.boundary['maxY'])
                z_range = (cnf.boundary['minZ'],cnf.boundary['maxZ'])

                im_size = (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)

                #convert x, y, w, z from pixel space to global space
                _x = x_range[0] + (x_range[1]-x_range[0]) * _x / im_size[0]
                _y = y_range[0] + (y_range[1]-y_range[0]) * _y / im_size[1]
                _l = (x_range[1]-x_range[0]) * _l / im_size[0]
                _w = (y_range[1]-y_range[0]) * _w / im_size[1]

                #convert  z, not in pixel space, but relative to the z range (z= 0 for minZ)
                _z = z_range[0] + _z
                _h = _h # h already oké
                #create carla object
                loc = carla.Location(float(_x),float(_y),float(_z))
                loc = lidar_transform.transform(loc)

                extent = carla.Vector3D(float(_l/2), float(_w/2), float(_h/2))
                bb = carla.BoundingBox(loc, extent)
                bb.rotation = carla.Rotation(pitch=0.0, yaw = float(np.degrees(-_yaw)), roll = 0.0)
                bb.rotation.pitch   += lidar_transform.rotation.pitch
                bb.rotation.yaw     += lidar_transform.rotation.yaw
                bb.rotation.roll    += lidar_transform.rotation.roll
                boxes.append(bb)

                #debug:
                if self.debug:
                    self._carlaWorld.world.debug.draw_box(bb, bb.rotation,  life_time=0.11,thickness=0.05,
                                                          color=carla.Color(g=0, r=0, b=200))

        if self.debug:
            #debug player bb:
            own = np.array([1, 152, 304, 1.0, 1, 1, 1, lidar_transform.rotation.yaw])
            _score, _x, _y, _z, _h, _w, _l, _yaw = own
            # from pixel coordinates to global coordinates
            # Todo: find this convertion !, check if below is correct
            x_range = (cnf.boundary['minX'], cnf.boundary['maxX'])
            y_range = (cnf.boundary['minY'], cnf.boundary['maxY'])
            z_range = (cnf.boundary['minZ'], cnf.boundary['maxZ'])

            im_size = (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)

            # convert x, y, w, z from pixel space to global space
            _x = x_range[0] + (x_range[1] - x_range[0]) * _x / im_size[0]
            _y = y_range[0] + (y_range[1] - y_range[0]) * _y / im_size[1]
            _l = (x_range[1] - x_range[0]) * _l / im_size[0]
            _w = (y_range[1] - y_range[0]) * _w / im_size[1]

            # convert  z, not in pixel space, but relative to the z range (z= 0 for minZ)
            _z = z_range[0] + _z
            _h = _h  # h already oké
            # create carla object
            loc = carla.Location(float(_x), float(_y), float(_z))
            lidar_transform.transform(loc)

            extent = carla.Vector3D(float(_l/2), float(_w/2), float(_h/2))
            bb = carla.BoundingBox(loc, extent)
            bb.rotation = carla.Rotation(pitch=0.0, yaw=float(_yaw), roll=0.0)
            bb.rotation.pitch += lidar_transform.rotation.pitch
            bb.rotation.yaw += lidar_transform.rotation.yaw
            bb.rotation.roll += lidar_transform.rotation.roll

            # debug:
            self._carlaWorld.world.debug.draw_box(bb, bb.rotation, life_time=0.1,thickness=0.05, color=carla.Color(g=0, r=0, b=200))
        return boxes

