import os
from collections import deque
import random

import torch
from torch import nn
import math
import numpy as np
import Environment

import matplotlib
import matplotlib.pyplot as plt
import wandb


ENABLE_WANDB = True
def _init_weights(m):
    if isinstance(m, nn.Linear):
       torch.nn.init.xavier_uniform_(m.weight)

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
            torch.nn.BatchNorm1d(self._hidden),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(self._hidden, self.num_actions),
            torch.nn.BatchNorm1d(self.num_actions),
            torch.nn.Tanh(),
        )
        self.var = torch.nn.Sequential(
            torch.nn.Linear(self._hidden, self.num_actions),
            torch.nn.BatchNorm1d(self.num_actions),
            torch.nn.Sigmoid(),
        )
        self.base.apply(_init_weights)
        self.mu.apply(_init_weights)
        self.var.apply(_init_weights)

    def forward(self, x):
        base_out = self.base(x)
        mu = self.mu(base_out)

        var = self.var(base_out)
        var = var + 1e-6 #avoid zero variance
        return torch.distributions.Normal(mu, var)

    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            dist = self.forward(state)
            action = dist.sample().detach()
        #action = action.clip(min=-1,max=1)
        #print(action)
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
            torch.nn.BatchNorm1d(self._hidden),
            torch.nn.ReLU(),
        )
        self.value = torch.nn.Linear(self._hidden, 1)

    def forward(self, x):
        x = self.base(x)
        return self.value(x)


class _Buffer:
    def __init__(self, capacity):
        """
        Parameters
        ----------
        capacity: int
            the length of your buffer
        """
        self.buffer = deque()


    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def get(self, batch_size):
        """
        batch_size: int
        """

        state, action, reward, next_state, done = zip(*self.buffer)
        self.buffer.clear()
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def clear(self):
        self.buffer.clear()

    def isFull(self):
        return False
        #return len(self) == self.buffer.maxlen

    def __len__(self):
        return len(self.buffer)



class ActorCriticLearner:
    def __init__(self, environment, config) -> None:
        self.config = config

        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.env = environment

        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config).to(self.device)

        # training parameters
        self.batch_size = config.get('batch_size', 32)
        self.num_frames = config.get('num_frames', 20000)
        self.gamma = config.get('gamma', 0.99)
        self.entropy_beta = config.get('entropy_beta',0.001)

        # network training:
        self.__actor_optim = torch.optim.Adam(self.actor.parameters())
        self.__critic_optim = torch.optim.Adam(self.critic.parameters())
        self.__loss_function = torch.nn.MSELoss()

        self.__buffer = _Buffer(config.get('batch_size', 256))


    def __update(self):
        if (len(self.__buffer) < 3):
            self.__buffer.clear()
            return
        self.actor.train()
        state, action, reward, next_state, done = self.__buffer.get(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.stack(action).squeeze(-1).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.BoolTensor(done).to(self.device)

        td_target = reward + self.gamma*self.critic(next_state).squeeze()*(done == False).int()
        value = self.critic(state).squeeze()
        advantage = td_target - value

        #actor
        norm_dists = self.actor(state)
        logs_probs = norm_dists.log_prob(action)
        entropy = norm_dists.entropy().mean()

        actor_loss = (-logs_probs * advantage.detach()).mean() - entropy * self.entropy_beta
        self.__actor_optim.zero_grad()
        actor_loss.backward()
        self.__actor_optim.step()
        #critic
        critic_loss = torch.nn.functional.mse_loss(td_target, value)
        self.__critic_optim.zero_grad()
        critic_loss.backward()
        self.__critic_optim.step()

        return actor_loss.cpu(),critic_loss.cpu()

    def train(self):
        self.initwandb()
        episode_reward = 0
        episode_count = 0
        losses = []
        all_rewards = []

        state = self.env.reset()
        log = dict()

        #a2c loss variabels
        advantages = []

        for frame_idx in range(1, self.num_frames + 1):
            self.actor.eval()
            action = self.actor.act(state)
            next_state, reward, done, info = self.env.step(action.item())
            if ENABLE_WANDB:
                wandb.log({'action':action.item(), 'reward':reward})
            #update
            self.__buffer.push(state,action,reward,next_state, done)
            if (self.__buffer.isFull() or done):
                actor_loss, critic_loss = self.__update()
                if ENABLE_WANDB:
                    wandb.log({'actor_loss' : actor_loss, 'critic_loss' : critic_loss})
                # loss = loss.data.cpu().numpy().tolist()
                # losses.append(loss)

            state = next_state
            episode_reward = reward + episode_reward * self.gamma

            if done:
                self.env.plot()
                state = self.env.reset()
                all_rewards.append(episode_reward)
                if ENABLE_WANDB:
                    wandb.log({'episode_reward':episode_reward})
                print(f'Episode reward:{episode_reward}')
                print(f'Frame count:{frame_idx}')
                episode_reward = 0
                episode_count += 1
            if frame_idx % 2000 == 0:
                # plot if required
                print(f"Current episode:{episode_reward}")
                print(f'Current state:{state}')
                plt.plot(all_rewards)
                plt.title("Episode rewards")
                plt.show()
                pass
            if frame_idx % 4000 == 0:
                self.save(f"{frame_idx}.pt")

    def initwandb(self):
        if not ENABLE_WANDB:
            return
        os.environ["WANDB_API_KEY"] ='827fc9095ed2096f0d61efa2cca1450526099892'

        wandb.login()
        wandb.init(project="a2c",config=self.config)


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
        self.actor.eval()
        while True:
            with torch.no_grad():
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
        'replay_size':10000,

        'num_inputs':6, #=size of states!
        'num_actions':1,
        'hidden':128,
        'entropy_beta':0.001,

        'continous_actions':True,
    }


    Environment._DISCRETE = False
    env = Environment.Environment(config)
    a2cLearner = ActorCriticLearner(env, config)
    #a2cLearner.load('working.pt')
    #a2cLearner.eval()

    a2cLearner.train()
    a2cLearner.save("TrainedModel.save")