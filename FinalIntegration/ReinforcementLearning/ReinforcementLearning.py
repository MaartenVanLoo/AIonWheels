import os
import time
from collections import deque

from .Qlearner import DQN

import torch
import numpy as np


class ReinforcementLearning(object):
    def __init__(self, carlaWorld, config=None) -> None:
        if not config:
            config = dict()

        super().__init__()
        self.config = config
        self._carlaWorld = carlaWorld
        self._agent = carlaWorld.getPlayer()

        self.device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self._model = DQN(self.config).to(self.device)
        self._model.eval() #set model in evaluation mode (not training mode)

        # if config contains a saved filepath, load this model
        path = self.config.get("model_path", "")
        if os.path.exists(path) and os.path.isfile(path):
            self.__loadModel(path)

        self.prev_action = 0
        self.frames = deque(maxlen=self.config.get('history_frames', 3))
        while len(self.frames) < self.config.get('history_frames', 3):
            self.frames.append(self.__getState(100))

    def getAction(self, distance):
        start = time.time()
        distance = np.clip(distance,0,100)
        self.frames.append(self.__getState(distance))
        state = np.array(list(self.frames)).flatten()
        action = self._model.act(state)
        self.prev_action = action
        stop = time.time()
        print(f"Inference time RL: {(stop - start)*1000:3.0f} ms")
        return action

    def __loadModel(self, filename) -> None:
        print(f"Loading RL model: {filename}")
        self._model.load_state_dict(torch.load(filename))
        print(f"Loading RL model done")

    def __getState(self, distance):
        return distance, self._agent.getTargetSpeed(), self._agent.getSpeed(), self.prev_action
        pass
