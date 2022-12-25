import os
import time

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

class DeeplearningLidar(object):
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
        #TODO: load trained model
        self._model_name = "/"
        self._model = self.create_model(self.config).to(self.device)
        self._model.eval() #set model in evaluation mode
        self.bev_map = None
        self.bev_image = None

        self.num_classes = self.config.get('num_classes')
        self.colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
          [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]

        self.distance = 110
        self.detected_boxes = None

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

    def getModelName(self)->str:
        return self._model_name

    def detect(self):
        start = time.time()
        sensor = self._carlaWorld.get_sensor("Lidar")
        if sensor is None:
            print(f"Inference time DL Lidar:\t{(time.time() - start) * 1000:4.0f} ms")
            return self.distance #previous value
        lidarData = sensor.getState()
        if lidarData is None:
            print(f"Inference time DL Lidar:\t{(time.time() - start) * 1000:4.0f} ms")
            return self.distance #no valid sensor state found

        lidarData = get_filtered_lidar(lidarData, cnf.boundary)
        self.bev_map = makeBEVMap(lidarData, cnf, cnf.boundary)
        self.bev_image = (self.bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
        with torch.no_grad():
            torch_map = torch.from_numpy(self.bev_map).to(self.device)
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

            self.bev_image = self._draw_output_map(self.bev_image, detections)
            self.detected_boxes = self._create_bounding_boxes(detections)
        print(f"Inference time DL Lidar:\t\t{(time.time() - start) * 1000:4.0f} ms")

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