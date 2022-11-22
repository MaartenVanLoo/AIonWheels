import random
from collections import deque

import torch
import math
import numpy as np
from Environment import Environment


class DQN():
    pass


class _ReplayBuffer:
    def __init__(self, capacity):
        """
        Parameters
        ----------
        capacity: int
            the length of your buffer
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        batch_size: int
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class Qlearner:
    #TODO: add replaybuffer
    def __init__(self, environment, DQN, gpu=True) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

        self.env = environment
        self.targetDQN = DQN().to(self.device)
        self.currentDQN = DQN().to(self.device)

        # training parameters
        self.batch_size = 32
        self.num_frames = 20000
        self.gamma = 0.99

        # network training:
        self.optimizer = torch.optim.Adam(self.currentDQN.parameters())
        self.loss_function = torch.nn.MSELoss()

    def train(self):
        episode_reward = 0
        for frame_idx in range(1, self.num_frames + 1):
            epsilon = self.__epsilon_by_frame(frame_idx)
            action = self.currentDQN.act(state, epsilon)

            next_state, reward, done, info = self.env.step(action);

            loss = self.__computeLoss(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                episode_reward = 0
            if frame_idx % 200 == 0:
                pass  # plot if required
            # update target every 100 frames
            if frame_idx % 100 == 0:
                self.__update_target()
        pass

    def save(self, filename: str):
        pass

    def load (self, filename: str):
        pass

    def __epsilon_by_frame(self, frame):
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame / epsilon_decay)

    def __update_target(self):
        self.targetDQN.load_state_dict(self.currentDQN.state_dict())

    def __computeLoss(self, state, action, reward, next_state, done):
        q_values = self.currentDQN.forward(state)
        next_q_values = self.targetDQN.forward(next_state)

        q_value = q_values[torch.arange(q_values.size(0)), action]

        next_q_value = next_q_values.max(dim=1).values * (done == False).int()
        expected_q_value = reward + torch.mul(next_q_value, self.gamma)

        loss = self.loss_function(q_value.to(torch.float64), expected_q_value.to(torch.float64))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


if __name__ == "__main__":
    env = Environment()
    qlearning = Qlearner(env, DQN, True)
    qlearning.train()
