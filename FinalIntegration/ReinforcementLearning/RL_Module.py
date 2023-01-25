import logging
import os
import time
from collections import deque

from .DQN import DQN

import torch
import numpy as np


class RL_Module(object):
    def __init__(self, carlaWorld, config=None) -> None:
        if not config:
            config = dict()
        super().__init__()
        # avoid speed limits over the actual limit!
        self.fail_safe = True


        self.config = config
        self._carlaWorld = carlaWorld
        self._agent = carlaWorld.getPlayer()

        self.device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self._model = DQN(self.config).to(self.device)
        self._model.eval() #set model in evaluation mode (not training mode)
        self._model_name = "/"

        # if config contains a saved filepath, load this model
        path = self.config.get("model_path", "")
        self.__loadModel(path)

        self.prev_action = 0
        self.frames = deque(maxlen=self.config.get('history_frames', 3))
        while len(self.frames) < self.config.get('history_frames', 3):
            self.frames.append(self.__getState(100))




    def getAction(self, distance, speed_limit = 999.9, red = False, orange= False, green = True):
        #adjust speed_limit based on lights, detection not reliable enough to directly input in RL agent:
        #if red:
        #    speed_limit = 0
        #elif orange:
        #    speed_limit /= 2

        start = time.time()
        distance = np.clip(distance,0,100)
        self.frames.append(self.__getState(distance, speed_limit))
        state = np.array(list(self.frames)).flatten()
        action = self._model.act(state)
        self.prev_action = action
        stop = time.time()
        print(f"Inference time RL:\t\t\t\t{(stop - start)*1000:4.0f} ms")
        return action

    def __loadModel(self, filename) -> None:
        if os.path.exists(filename) and os.path.isfile(filename):
            _, self._model_name = os.path.split(filename)
            print(f"Loading RL model: {self._model_name}")
            self._model.load_state_dict(torch.load(filename))
            print(f"Loading RL model: {self._model_name} done")
        else:
            print(f"Could not load RL model")

    def __getState(self, distance, speed_limit = 999.9):
        return distance, self._agent.getTargetSpeed(), self._agent.getSpeed(), self.prev_action
        # code below could use inputted speed limit but not reliable enough to use:
        #if (self.fail_safe):
        #    return distance, min(self._agent.getTargetSpeed(), speed_limit), self._agent.getSpeed(), self.prev_action
        #else:
        #    return distance, speed_limit, self._agent.getSpeed(), self.prev_action
        #pass


    def getModelName(self)->str:
        return self._model_name