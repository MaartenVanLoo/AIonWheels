import time

from data_process.kitti_bev_utils import makeBEVMap, get_filtered_lidar
import torch

import FinalIntegration.DeepLearningLidar.config.kitti_config as cnf
import numpy as np
from models.model_utils import create_model

class DeeplearningLidar(object):
    def __init__(self, carlaWorld, config = None) -> None:
        if not config:
            config = dict()
        super().__init__()
        self.config = config
        self._carlaWorld = carlaWorld
        self._agent = carlaWorld.getPlayer()

        self.device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        #TODO: aaanvullen
        self._model = create_model(config)
        #self._model.eval() #set model in evaluation mode
        self._model_name = "/"
        self.bev_map = None
        self.bev_image = None

        self.distance = 110

    def getModelName(self)->str:
        return self._model_name

    def getDistance(self):
        start = time.time()
        sensor = self._carlaWorld.get_sensor("Lidar")
        if sensor is None:
            print(f"Inference time DL Lidar:\t{(time.time() - start) * 1000:3.0f} ms")
            return self.distance #previous value
        lidarData = sensor.getState()
        if lidarData is None:
            print(f"Inference time DL Lidar:\t{(time.time() - start) * 1000:3.0f} ms")
            return self.distance #no valid sensor state found

        lidarData = get_filtered_lidar(lidarData, cnf.boundary)
        self.bev_map = makeBEVMap(lidarData, cnf, cnf.boundary)
        self.bev_image = (self.bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)

        #Todo: forward model
        output = self._model(self.bev_map)
        #Todo: decode result
        #result = decode(output)

        print(f"Inference time DL Lidar:\t{(time.time() - start) * 1000:3.0f} ms")

    def getBoundingBoxes(self):
        return 0
