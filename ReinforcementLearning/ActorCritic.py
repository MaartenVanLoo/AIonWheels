from collections import deque
import random

import torch
from torch import nn
import math
import numpy as np
from Environment import Environment
import matplotlib
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

        self.num_inputs = config.get('num_inputs', 3)
        self._hidden = config.get('hidden', 128)
        if isinstance(self._hidden,list):
            self._hidden = 128 if (len(self._hidden)== 0) else self._hidden[0]
        self.num_actions = config.get('num_actions', 7)

        self.base = torch.nn.Sequential(
            torch.nn.Linear(self.num_inputs, self._hidden),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(self._hidden, self.num_actions),
            torch.nn.Tanh()
        )
        self.var = torch.nn.Sequential(
            torch.nn.Linear(self._hidden, self.num_actions),
            torch.nn.Softplus()
        )
    def forward(self, x):
        x = self.base(x)
        mu = self.mu(x)
        var = self.var(x)
        return torch.distributions.Normal(mu, var)
    def act(self,state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.forward(state)
        action = action.sample().detach()
        action = torch.clip(action,-1,1)
        return action

class Critic(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

        self.num_inputs = config.get('num_inputs', 3)
        self._hidden = config.get('hidden', 128)
        if isinstance(self._hidden,list):
            self._hidden = 128 if (len(self._hidden)== 0) else self._hidden[0]
        self.num_actions = config.get('num_actions', 7)

        self.base = torch.nn.Sequential(
            torch.nn.Linear(self.num_inputs, self._hidden),
            torch.nn.ReLU(),
        )
        self.value = torch.nn.Linear(self._hidden, 1)

    def forward(self, x):
        x = self.base(x)
        return self.value(x)


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



class ActorCriticLearner:
    def __init__(self, environment, config) -> None:
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.env = environment

        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config).to(self.device)

        # training parameters
        self.batch_size = config.get('batch_size', 32)
        self.num_frames = config.get('num_frames', 20000)
        self.gamma = config.get('gamma', 0.99)
        self.entropy_beta = config.get('entropy_beta',0)

        # network training:
        self.__actor_optim = torch.optim.Adam(self.actor.parameters())
        self.__critic_optim = torch.optim.Adam(self.critic.parameters())
        self.__loss_function = torch.nn.MSELoss()

        self.__replay_buffer = _ReplayBuffer(config.get('replay_size', 50000))


    def __update(self):
        state, action, reward, next_state, done = self.__replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.stack(action).to(self.device) #action is already a tensor
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        td_target = reward + self.gamma*self.critic(next_state).squeeze()*(1-done)
        value = self.critic(state).squeeze()
        advantage = td_target - value

        #actor
        norm_dists = self.actor(state)
        logs_probs = norm_dists.log_prob(action)
        entropy = norm_dists.entropy().mean()

        actor_loss = (-logs_probs * advantage.detach()).mean() - entropy * self.entropy_beta
        self.__actor_optim.zero_grad()
        actor_loss.backward()

        #critic
        critic_loss = torch.nn.functional.mse_loss(td_target, value)
        self.__critic_optim.zero_grad()
        critic_loss.backward()

        return actor_loss,critic_loss
        pass

    def train(self):
        episode_reward = 0
        losses = []
        all_rewards = []

        state = self.env.reset()
        for frame_idx in range(1, self.num_frames + 1):
            action = self.actor.act(state)

            next_state, reward, done, info = self.env.step(action.item())

            #print(env)
            #print(done)
            if not 'TimeLimit.truncated' in info.keys() or not info['TimeLimit.truncated']:
                self.__replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if (len(self.__replay_buffer) > self.batch_size):
                loss = self.__update()
                #loss = loss.data.cpu().numpy().tolist()
                #losses.append(loss)


            if done:
                if random.random() > 0.9:
                    self.env.plot()
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
            if frame_idx % 2000 == 0:
                # plot if required
                plt.plot(all_rewards)
                plt.title("Episode rewards")
                plt.show()
                pass
            if frame_idx % 4000 == 0:
                self.save(f"{frame_idx}.pt")

    def save(self, filename: str):
        # TODO: add support for saving a model
        torch.save(self.actor.state_dict(), "actor_"+filename)
        torch.save(self.critic.state_dict(), "critic_"+filename)
        pass

    def load(self, filename: str):
        # TODO: add support for loading a model
        self.actor.load_state_dict(torch.load("actor_"+filename))
        self.critic.load_state_dict(torch.load("critic_"+filename))
        pass

    def eval(self):
        episode_reward= 0
        state = self.env.reset()
        while True:
            action = self.actor.act(state)
            next_state, reward, done, info = self.env.step(action.item())
            state = next_state
            episode_reward += reward
            if done:
                self.env.plot()
                state = self.env.reset()
                print(f'Episode reward: {episode_reward}')
                break

if __name__ == "__main__":
    #config = optional, default values have been set in the qlearning framework
    config = {
        'device':'cuda',
        'batch_size':256,
        'num_frames':200000,
        'gamma':0.99,
        'replay_size':50000,

        'num_input':3, #=size of states!
        'num_actions':1,
        'hidden':128,

        'continous_actions':True,
    }


    env = Environment(config)
    a2cLearner = ActorCriticLearner(env, config)
    a2cLearner.load('working.pt')
    a2cLearner.eval()

    #a2cLearner.train()
    #a2cLearner.save("TrainedModel.save")