import operator
import os
import pathlib
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
ENABLE_WANDB=True
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
            print(f"Random Action: {self.prev_action}")
            return self.prev_action
        elif self.frame_idx < 0:
            #Take own policy:
            self.frame_idx += 1
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0].cpu().numpy().tolist()  # argmax over actions
            print(f"Policy Action: {action}")
            return action


        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0].cpu().numpy().tolist()  # argmax over actions
            self.frame_idx = -5
            print(f"Policy Action: {action}")
        else:
            self.frame_idx = 5
            action = random.randint(0, self.num_actions - 1)
            print(f"Random Action: {self.prev_action}")

        self.prev_action = action
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


# https://towardsdatascience.com/how-to-implement-prioritized-experience-replay-for-a-deep-q-network-a710beecd77b
class _PriorityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, experiences_per_sampling, seed, compute_weights,
                 config=dict()):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            experiences_per_sampling (int): number of experiences to sample during a sampling iteration
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experiences_per_sampling = experiences_per_sampling

        self.alpha = 0.5
        self.alpha_decay_rate = 0.99
        self.beta = 0.5
        self.beta_growth_rate = 1.001
        self.seed = random.seed(seed)
        self.compute_weights = compute_weights
        self.experience_count = 0

        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.data = namedtuple("Data",
                               field_names=["priority", "probability", "weight", "index"])

        indexes = []
        datas = []
        for i in range(buffer_size):
            indexes.append(i)
            d = self.data(0, 0, 0, i)
            datas.append(d)

        self.memory = {key: self.experience for key in indexes}
        self.memory_data = {key: data for key, data in zip(indexes, datas)}
        self.sampled_batches = []
        self.current_batch = 0
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1

    def update_priorities(self, tds, indices):
        for td, index in zip(tds, indices):
            N = min(self.experience_count, self.buffer_size)

            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority

            if self.compute_weights:
                updated_weight = ((N * updated_priority) ** (-self.beta)) / self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1

            old_priority = self.memory_data[index].priority
            self.priorities_sum_alpha += updated_priority ** self.alpha - old_priority ** self.alpha
            updated_probability = td[0] ** self.alpha / self.priorities_sum_alpha
            data = self.data(updated_priority, updated_probability, updated_weight, index)
            self.memory_data[index] = data

    def update_memory_sampling(self):
        """Randomly sample X batches of experiences from memory."""
        # X is the number of steps before updating memory
        self.current_batch = 0
        values = list(self.memory_data.values())
        random_values = random.choices(self.memory_data,
                                       [data.probability for data in values],
                                       k=self.experiences_per_sampling)
        self.sampled_batches = [random_values[i:i + self.batch_size]
                                for i in range(0, len(random_values), self.batch_size)]

    def update_parameters(self):
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1
        N = min(self.experience_count, self.buffer_size)
        self.priorities_sum_alpha = 0
        sum_prob_before = 0
        for element in self.memory_data.values():
            sum_prob_before += element.probability
            self.priorities_sum_alpha += element.priority ** self.alpha
        sum_prob_after = 0
        for element in self.memory_data.values():
            probability = element.priority ** self.alpha / self.priorities_sum_alpha
            sum_prob_after += probability
            weight = 1
            if self.compute_weights:
                weight = ((N * element.probability) ** (-self.beta)) / self.weights_max
            d = self.data(element.priority, probability, weight, element.index)
            self.memory_data[element.index] = d
        print("sum_prob before", sum_prob_before)
        print("sum_prob after : ", sum_prob_after)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.experience_count += 1
        index = self.experience_count % self.buffer_size

        if self.experience_count > self.buffer_size:
            temp = self.memory_data[index]
            self.priorities_sum_alpha -= temp.priority ** self.alpha
            if temp.priority == self.priorities_max:
                self.memory_data[index].priority = 0
                self.priorities_max = max(self.memory_data.items(), key=operator.itemgetter(1)).priority
            if self.compute_weights:
                if temp.weight == self.weights_max:
                    self.memory_data[index].weight = 0
                    self.weights_max = max(self.memory_data.items(), key=operator.itemgetter(2)).weight

        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self.alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha
        e = self.experience(state, action, reward, next_state, done)
        self.memory[index] = e
        d = self.data(priority, probability, weight, index)
        self.memory_data[index] = d

    def sample(self):
        sampled_batch = self.sampled_batches[self.current_batch]
        self.current_batch += 1
        experiences = []
        weights = []
        indices = []

        for data in sampled_batch:
            experiences.append(self.memory.get(data.index))
            weights.append(data.weight)
            indices.append(data.index)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, weights, indices)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Qlearner:
    def __init__(self, environment, DQN, config) -> None:
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.env = environment
        self.config=config

        self.targetDQN = DQN(config).to(self.device)
        self.currentDQN = DQN(config).to(self.device)

        # training parameters
        self.batch_size = config.get('batch_size', 32)
        self.mini_batch = config.get('mini_batch', 16)
        self.num_frames = config.get('num_frames', 20000)
        self.gamma = config.get('gamma', 0.99)

        # network training:
        self.__optimizer = torch.optim.Adam(self.currentDQN.parameters(), lr=config.get('lr', 0.00025))
        self.__loss_function = torch.nn.MSELoss()

        self.__replay_buffer = _ReplayBuffer(config.get('replay_size', 50000))

        #plotting:
        #self.__episode_rewards_fig = plt.figure()
        #self.__episode_rewards_ax = self.__episode_rewards_fig.add_subplot(1,1,1)

        self.wandb_enabled = False
        self.model_name = "TrainedModel.pth"
        self.initwandb()



    def save(self,filename: str)->None:
        path = pathlib.Path(filename)
        if not path.parent.exists():
            os.makedirs(path.parent)
        torch.save(self.currentDQN.state_dict(), filename)
        pass

    def load(self, filename: str)->None:
        self.currentDQN.load_state_dict(torch.load(filename))
        self.targetDQN.load_state_dict(torch.load(filename))
        pass

    def __update_target(self):
        self.targetDQN.load_state_dict(self.currentDQN.state_dict())
        #soft update: TODO: CODE NOT CHECKED! verify if this code works
        #tau = 1e-3
        #for target_param, local_param in zip(self.targetDQN.parameters(), self.currentDQN.parameters()):
        #    target_param.data.copy_(tau*local_param.data + (1-tau) * target_param.data)

    def __epsilon_by_frame(self, frame):
        epsilon_start = self.config.get('epsilon_start',0.1)
        epsilon_final = self.config.get('epsilon_final',0.002)
        epsilon_decay = self.config.get('epsilon_decay',10000)
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame / epsilon_decay)

    def __update(self):
        self.targetDQN.train()
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
        torch.nn.utils.clip_grad_norm_(self.currentDQN.parameters(), 1, error_if_nonfinite=True)
        self.__optimizer.step()
        return loss

    def train(self):
        collisions = 0
        episode_reward = 0
        losses = []
        all_rewards = []
        movingAverage = []

        state = self.env.reset()
        n_experiences = 0
        for frame_idx in tqdm(range(1, self.num_frames + 1)):
            self.metrics = {}
            self.currentDQN.eval()
            epsilon = self.__epsilon_by_frame(frame_idx)
            self.metrics['epsilon']=epsilon
            action = self.currentDQN.act(state, epsilon)
            self.metrics['action']=action

            next_state, reward, done, info = self.env.step(action)
            self.metrics['reward']=reward

            if 'collision' in info.keys():
                collisions += int(info['collision'])
                self.metrics['collision'] = collisions

            if not 'TimeLimit.truncated' in info.keys() or not info['TimeLimit.truncated']:
                self.__replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            n_experiences += 1
            if len(self.__replay_buffer) > self.batch_size and n_experiences >= self.mini_batch:
                loss = self.__update()
                self.metrics['loss']=loss

                n_experiences = 0
                # loss = loss.data.cpu().numpy().tolist()
                # losses.append(loss)

            if done:
                if random.random()>0.5: # avoid plotting everything
                    self.env.plot()
                    pass
                self.metrics['episode_length'] = self.env.stepCount
                state = self.env.reset()
                all_rewards.append(episode_reward)
                movingAverage.append(sum(all_rewards[-_mov_average_size:]) / min(_mov_average_size, len(all_rewards)))
                print(f'Finished episode:')
                print(f'Episode reward:{episode_reward}')
                print(f'mov avg reward:{movingAverage[-1]}')
                print(f'Max reward:{max(all_rewards)}')
                print(f'Frame count:{frame_idx}')
                print()
                self.metrics['episode_reward'] = episode_reward
                self.metrics['mov_avg_episode_reward']=movingAverage[-1]
                self.metrics['episode_count'] = len(all_rewards)
                episode_reward = 0
            if frame_idx % 100000 == 0:
                # plot if required
                print()
                print(f"Current episode:{episode_reward}")
                print(f'Current state:{state}')
                print()
                #self.__episode_rewards_ax.cla()
                #self.__episode_rewards_ax.plot(all_rewards, color='blue')
                #self.__episode_rewards_ax.plot(movingAverage, color='red')
                #self.__episode_rewards_ax.set_title("Episode rewards")
                #self.__episode_rewards_fig.canvas.draw()
                #self.__episode_rewards_fig.canvas.flush_events()
                plt.plot(all_rewards, color='blue')
                plt.plot(movingAverage, color='red')
                plt.title("Episode rewards")
                plt.show(block = False)
                pass
            # update target every 1000 frames
            if frame_idx % self.config.get('target_update_freq',20000) == 0:
                self.__update_target()
            # save network every 20 000 frames
            if frame_idx % 100000 == 0:
                self.save(f"DQN_{frame_idx}.pt")

            if (self.wandb_enabled):
                wandb.log(self.metrics)
                self.metrics={}
        pass

    def eval(self):
        state = self.env.reset()
        episode_reward = 0.0
        episode_rewards = [0.0]
        moving_averages = [0.0]

        # set evaluation mode
        self.currentDQN.eval()
        self.env.eval()
        while True:
            with torch.no_grad():
                action = self.currentDQN.act(state)
                self.env.step(action)
                next_state, reward, done, info = self.env.step(action)

                state = next_state
                episode_reward += reward
                episode_rewards.append(episode_reward)
                moving_averages.append(sum(episode_rewards[-_mov_average_size:]) / min(_mov_average_size,
                                                                                       len(episode_rewards)))
                if done:
                    break
        # done evaluating
        print()
        print(f'Finished final episode:')
        print(f'Episode reward:{episode_rewards[-1]}')
        print(f'mov avg reward:{moving_averages[-1]}')
        self.env.plot()
        #self.__episode_rewards_ax.cla()
        #self.__episode_rewards_ax.plot(episode_rewards, color='blue')
        #self.__episode_rewards_ax.plot(moving_averages, color='red')
        #self.__episode_rewards_ax.set_title("Episode rewards")
        #self.__episode_rewards_fig.canvas.draw()
        #self.__episode_rewards_fig.canvas.flush_events()
        fig, ax1= plt.subplots(1, 1)
        ax1.plot(episode_rewards, color='blue')
        ax1.plot(moving_averages, color='red')
        ax1.set_title("Episode rewards")
        plt.show(block = False)
        self.env.train()  # set back to training mode


    def initwandb(self):
        if not ENABLE_WANDB:
            return
        os.environ["WANDB_API_KEY"] ='827fc9095ed2096f0d61efa2cca1450526099892'

        wandb.login()
        run = wandb.init(project="AIonWheels_RL", tags="qLearning",config=self.config)
        self.wandb_enabled = True
        self.model_name = run.name + ".pth"

