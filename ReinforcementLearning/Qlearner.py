import random
from collections import deque

import torch
import math
import numpy as np
from Environment import Environment
import matplotlib
import matplotlib.pyplot as plt

class DQN(torch.nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.device = torch.device(config.get('device','cuda') if torch.cuda.is_available() else 'cpu')
        self.num_inputs=config.get('num_inputs',2)
        self._hidden = config.get('hidden',[128, 128])
        self.num_actions=config.get('num_actions',7)

        modules =[]
        modules.append(torch.nn.Linear(self.num_inputs, self._hidden[0]))
        modules.append(torch.nn.ReLU())
        for i in range(0,len(self._hidden)-1):
            modules.append(torch.nn.Linear(self._hidden[i], self._hidden[i+1]))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(self._hidden[len(self._hidden)-1], self.num_actions))

        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0].cpu().numpy().tolist()  # argmax over actions
        else:
            action = random.randint(0, self.num_actions - 1)
        return action

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
    def __init__(self, environment, DQN, config) -> None:
        self.device = torch.device(config.get('device','cuda') if torch.cuda.is_available() else 'cpu')

        self.env = environment
        self.targetDQN = DQN(config).to(self.device)
        self.currentDQN = DQN(config).to(self.device)

        # training parameters
        self.batch_size = config.get('batch_size',32)
        self.num_frames = config.get('num_frames', 20000)
        self.gamma = config.get('gamma',0.99)

        # network training:
        self.__optimizer = torch.optim.Adam(self.currentDQN.parameters())
        self.__loss_function = torch.nn.MSELoss()

        self.__replay_buffer = _ReplayBuffer(config.get('replay_size',50000))

    def train(self):
        episode_reward = 0
        losses = []
        all_rewards = []

        state = self.env.reset()
        for frame_idx in range(1, self.num_frames + 1):
            epsilon = self.__epsilon_by_frame(frame_idx)
            action = self.currentDQN.act(state, epsilon)

            next_state, reward, done, info = self.env.step(action);

            print(env)
            print(done)
            if not 'TimeLimit.truncated' in info.keys() or not info['TimeLimit.truncated']:
                self.__replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if (len(self.__replay_buffer) > self.batch_size):
                loss = self.__computeLoss()
                #loss = loss.data.cpu().numpy().tolist()
                #losses.append(loss)



            if done:
                env.plot()
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
            if frame_idx % 2000 == 0:
                # plot if required
                plt.plot(all_rewards)
                plt.show()
                pass
            # update target every 100 frames
            if frame_idx % 100 == 0:
                self.__update_target()
        pass

    def save(self, filename: str):
        #TODO: add support for saving a model
        pass

    def load (self, filename: str):
        #TODO: add support for loading a model
        pass

    def __epsilon_by_frame(self, frame):
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame / epsilon_decay)

    def __update_target(self):
        self.targetDQN.load_state_dict(self.currentDQN.state_dict())

    def __computeLoss(self):
        state, action, reward, next_state, done = self.__replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.BoolTensor(done).to(self.device)


        q_values = self.currentDQN.forward(state)
        next_q_values = self.targetDQN.forward(next_state)

        q_value = q_values[torch.arange(q_values.size(0)), action]

        next_q_value = next_q_values.max(dim=1).values * (done == False).int()
        expected_q_value = reward + torch.mul(next_q_value, self.gamma)

        loss = self.__loss_function(q_value.to(torch.float64), expected_q_value.to(torch.float64))

        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
        return loss


if __name__ == "__main__":
    #config = optional, default values have been set in the qlearning framework
    config = {
        'device':'cuda',
        'batch_size':64,
        'num_frames':200000,
        'gamma':0.99,
        'replay_size':50000,

        'num_input':2, #=size of states!
        'num_actions':71,
        'hidden':[128,128],
    }
    env = Environment(config)
    qlearning = Qlearner(env, DQN, config)
    qlearning.train()
    qlearning.save("TrainedModel.save")
