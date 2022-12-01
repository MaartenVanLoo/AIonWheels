import argparse
import math
import os
import traceback
from collections import deque

from CarlaQlearning import CarlaConnection
import numpy as np

from ReinforcementLearning.Qlearner import Qlearner, DQN

_DISCRETE = True


def _sigmoid(x):
    if (x < -512):  # avoid overflow of 'exp' function
        return 0
    return 1 / (1 + math.exp(-x))


class CarlaRLEnv(CarlaConnection):
    def __init__(self, config, args, host='127.0.0.1', port=2000, width=1280, height=720, fullscreen = False,
                 show=True) -> None:
        super().__init__(config)
        self.target_speed = 5
        self.prev_action = 0
        self.frames = deque(maxlen=config.get('history_frames', 3))
        self.num_actions = config.get('num_actions', 7)
        self.reward_offset = config.get('reward_offset', 1.5)
        self.t_gap = 2
        self.distance_default = 4

        self.target_speed=config.get('target_speed',15)
        self.config = config

        self.episodeReward=0
        self.stepCount = 0

        self.train()

        #compute total history size:
        self.history_size =self.frames.maxlen * 4 + 2 # state, reward, done
        self.episodeHistory = []


    def render(self):
        pass
    def plot(self):
        pass

    def reset(self):
        super().reset()
        self.step(int((self.num_actions + 1) / 2))

        distance = self.get_distance()
        self.frames = deque(maxlen=self.config.get('history_frames', 3))
        while len(self.frames) < self.config.get('history_frames', 3):
            self.frames.append(self.__getState(distance))


        self.episodeReward=0
        self.stepCount = 0

        state = np.array(list(self.frames)).flatten()
        #save prev episode if not empty
        if len(self.episodeHistory) > 5:
            with open("../CarlaEpisodeData/counter","r") as counter:
                count = int(counter.read())
            count += 1
            with open("../CarlaEpisodeData/counter","w") as counter:
                counter.write(str(count))
            filename = "../CarlaEpisodeData/"+str(count).zfill(6) + "_episode_data.data"
            np.save(filename,np.asarray(self.episodeHistory))
        self.episodeHistory = []
        return state

    def step(self, action: int):
        # step state
        control = super().step(action)
        self.stepCount += 1

        distance = self.get_distance()
        self.frames.append(self.__getState(distance))
        state = np.array(list(self.frames)).flatten()

        # reward
        reward = self.__getReward(action, distance)

        # done
        done = self.__isTerminal(distance)

        # info
        info = self.__info()

        self.episodeReward += reward
        self.prev_action = action
        if self.stepCount%100==0:
            print(f"Frame:{self.stepCount}")
            print(state)
        self.episodeHistory.append(np.concatenate((state , np.array([reward, done]))))
        return state, reward, done, info

    def eval(self):
        #set environment in evaluation mode:
        self.evaluation = True
    def train(self):
        self.evaluation = False

    def __getState(self, distance):
        distance = np.clip(distance, 0, 500)
        return distance, self.target_speed, self.agent.getVel().length(), self.prev_action

    def __getReward(self, action, distance):

        distance_penalty = 2 if (distance < self.__getSafeDistance()) else 0

        target_speed = self.__getTargetSpeed(distance)
        e = target_speed - self.agent.getVel().length()
        m_t = 1 if e * e <= 0.25 else 0

        # u = deviation from previous action => encourage low changes in speed
        if _DISCRETE:
            u = self.agent.actions[action] - self.agent.actions[self.prev_action]
        else:
            u = action - self.prev_action
        # https://nl.mathworks.com/help/reinforcement-learning/ug/train-ddpg-agent
        # -for-adaptive-cruise-control.html
        reward = -(0.1 * e * e + u * u) + m_t - distance_penalty + self.reward_offset
        # speed_reward = -sigmoid(abs(dv/5))*2+2

        # combined_reward = distance_reward*sigmoid(-distance+10)+\
        #                  speed_reward*sigmoid(distance - 10)
        reward = max(reward, -10)
        return reward

    def __isTerminal(self, distance):
        #check for collision:

        #is terminal
        if self.evaluation:
            done = distance <= 0.5 or \
                   self.episodeReward > 50999 or \
                   self.stepCount > 50000
            # (distance > 1000 and self.agent.getSpeed() >= self.car.getSpeed()) or \
        else:
            done = distance <= 0.5 or \
                   self.episodeReward > 9999 or \
                   self.stepCount > 5000
            # (distance > 1000 and self.agent.getSpeed() >= self.car.getSpeed()) or \
        return self.stepCount > 10000

    def __info(self):
        return {} #empty dict


    def __getSafeDistance(self):
        return self.t_gap * abs(self.agent.getVel().length()) + self.distance_default

    def __getTargetSpeed(self, distance):
        if (distance == np.Inf):
            return self.target_speed
        save_distance = self.__getSafeDistance()
        lead_car_speed = self.get_speed_leading_car().length()

        # smooth transition of target speed:
        # v1 when far away, v2 when near
        # alpha = 1 when distance > save distance
        # alpha = -1 when distance < save distance
        v1 = self.target_speed
        v2 = min(lead_car_speed, self.target_speed)
        v = self.agent.getVel().length()
        speed = v if not abs(v) < 1e-6 else 1e-5
        speed = max(abs(speed), 4)

        alpha = _sigmoid((distance - save_distance) / abs(speed)) * 2 - 1
        target_speed = alpha * v1 + (1 - alpha) * v2

        return target_speed



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')

    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    config = {
        'device': 'cuda',
        'batch_size': 2048,
        'mini_batch': 24,  # only update once after n experiences
        'num_frames': 51000,
        'gamma': 0.90,
        'replay_size': 250000,
        'lr': 0.0003,
        'reward_offset': 1.5,
        #epsilon greedy
        'epsilon_start':0.5,
        'epsilon_final': 0.04,
        'epsilon_decay':3000,
        'target_update_freq':2000,
        'history_frames': 3,
        'num_inputs': 12,  # =size of states!
        'num_actions': 11,
        'hidden': [128, 512, 512, 128, 64],
    }

    worldapi=None
    Qlearner.ENABLE_WANDB = False
    try:
        worldapi = CarlaRLEnv(config=config, args=args)
        worldapi.init()
        print(worldapi.agent.getPos())

        #config['num_inputs'] = len(worldapi.reset())  # always correct, but expensive in a carla environent
        qlearning = Qlearner(worldapi, DQN, config)
        qlearning.load("../models/TrainedModel_meadow-63.pth") # works well
        #qlearning.load("../models/TrainedModel_sky-64.pth")
        #qlearning.load("../models/prime-sun-67.pth")
        #qlearning.train()
        qlearning.eval()
        #qlearning.save("../models/TrainedModel_meadow-63_carla.pth")
        #qlearning.save("../models/prime-sun-67_carla.pth")
        worldapi.reset() #force to save episode history
        #while True:
        #    print(worldapi.step(action=8))
    except:
        traceback.print_exc()
    finally:
        if worldapi:
            worldapi.cleanup()

    pass
