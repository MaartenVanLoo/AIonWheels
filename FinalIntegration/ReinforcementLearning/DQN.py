import operator
import os
import random
from bisect import bisect
from collections import deque, namedtuple

import torch
import math
import numpy as np
from tqdm import tqdm

import wandb

import matplotlib
import matplotlib.pyplot as plt

_mov_average_size = 10  # moving average of last 10 epsiodes
ENABLE_WANDB = False
#matplotlib.use("Tkagg")

class DQN(torch.nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.num_inputs = config.get('num_inputs', 3)
        self._hidden = config.get('hidden', [128, 128])
        self.num_actions = config.get('num_actions', 7)

        self.frame_idx = 0
        self.prev_action = None

        modules = []
        modules.append(torch.nn.Linear(self.num_inputs, self._hidden[0]))
        modules.append(torch.nn.ReLU())
        for i in range(0, len(self._hidden) - 1):
            modules.append(torch.nn.Linear(self._hidden[i], self._hidden[i + 1]))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(self._hidden[len(self._hidden) - 1], self.num_actions))

        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def act(self, state, epsilon=0.0):
        if self.frame_idx > 0:
            #Repeat random action
            self.frame_idx -= 1
            #print(f"Random Action: {self.prev_action}")
            return self.prev_action
        elif self.frame_idx < 0:
            #Take own policy:
            self.frame_idx += 1
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0].cpu().numpy().tolist()  # argmax over actions
            #print(f"Policy Action: {action}")
            return action


        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0].cpu().numpy().tolist()  # argmax over actions
            self.frame_idx = -5
            #print(f"Policy Action: {action}")
        else:
            self.frame_idx = 5
            action = random.randint(0, self.num_actions - 1)
            #print(f"Random Action: {self.prev_action}")

        self.prev_action = action
        return action

