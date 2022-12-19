import argparse
import math
import os
import traceback
from collections import deque


import numpy as np
import carla

from ReinforcementLearning.CarlaEnvironment.CarlaWorldFast import CarlaWorldFast
from ReinforcementLearning.CarlaEnvironment.utils.dist import distanceAlongPath
from ReinforcementLearning.Qlearner import Qlearner, DQN

_DISCRETE = True


def _sigmoid(x):
    if (x < -512):  # avoid overflow of 'exp' function
        return 0
    return 1 / (1 + math.exp(-x))


#class CarlaRLEnv(CarlaWorldAPI):
class CarlaRLEnvFast(CarlaWorldFast):
    def __init__(self, config, args) -> None:
        super().__init__(args)
        self.prev_action = 0
        self.frames = deque(maxlen=config.get('history_frames', 3))
        self.num_actions = config.get('num_actions', 7)
        self.reward_offset = config.get('reward_offset', 1.5)
        self.t_gap = 2
        self.distance_default = 4

        self.user_set_point=config.get('target_speed', 30)
        self.config = config

        self.episodeReward=0
        self.stepCount = 0

        self.train()

        #compute total history size:
        self.history_size =self.frames.maxlen * 4 + 2 # state, reward, done
        self.episodeHistory = []

        self.frame= 0
        self.world.on_tick(self.on_world_tick)

    def render(self):
        pass
    def plot(self):
        pass

    def reset(self,map=None, layers=carla.MapLayer.All):
        super().reset(map=None, layers=carla.MapLayer.Buildings | carla.MapLayer.Ground | carla.MapLayer.Foliage )

        #vehicleId, distance = self.getClosestVechicle()
        distance, _ = distanceAlongPath(
            self.getGlobalPath(),
            self.getGlobalBoundingBoxes(),
            self.getPlayer().getWidth(),
            self.world,
            debug=self.debug
        )
        self.frames = deque(maxlen=self.config.get('history_frames', 3))
        while len(self.frames) < self.config.get('history_frames', 3):
            self.frames.append(self.__getState(distance))


        self.episodeReward=0
        self.stepCount = 0

        state = np.array(list(self.frames)).flatten()
        #save prev episode if not empty
        if len(self.episodeHistory) > 5:
            #with open("../CarlaEpisodeData/counter","r") as counter:
            #    count = int(counter.read())
            #count += 1
            #with open("../CarlaEpisodeData/counter","w") as counter:
            #    counter.write(str(count))
            #filename = "../CarlaEpisodeData/"+str(count).zfill(6) + "_episode_data.data"
            #np.save(filename,np.asarray(self.episodeHistory))
            pass
        self.episodeHistory = []

        self.world.on_tick(self.on_world_tick)
        return state

    def on_world_tick(self, timestamp):
        self.frame = timestamp.frame_count

    def step(self, action: int):
        # step state
        distance, vehicleId = super().step(action)
        self.stepCount += 1

        #vehicleId, distance = self.getClosestVechicle()
        if self.debug:
            print(f"Distance:{distance}")
        self.frames.append(self.__getState(distance))
        state = np.array(list(self.frames)).flatten()

        # reward
        reward = self.__getReward(action, distance, vehicleId)

        # done
        done = self.__isTerminal(distance)

        # info
        info = self.__info(distance, vehicleId)

        self.episodeReward += reward
        self.prev_action = action
        #if self.stepCount%1==0:
        #    print(f"Frame:{self.stepCount}")
        #    print(self.getCollisionIntensity())
        #    print(state)
        self.episodeHistory.append(np.concatenate((state , np.array([reward, done]))))

        if self.debug:
            print(f"User Request: {self.user_set_point}. Speed limit: {self.agent.getSpeedLimit()}")
        return state, reward, done, info

    def eval(self):
        #set environment in evaluation mode:
        self.evaluation = True
    def train(self):
        self.evaluation = False


    def __getState(self, distance):
        distance = np.clip(distance, 0, 100)
        return distance, min(self.user_set_point, self.getPlayer().getSpeedLimit()), self.getPlayer().getSpeed(), \
                             self.prev_action

    def __getReward(self, action, distance, vehicleId):
        state = self.get_sensor("CollisionSensor").getState()
        if state is None:
            isCollision = False
        else:
            state_frame, state_vehicle = state
            if not state_frame == self.frame:
                isCollision = False
            else:
                isCollision = not state_vehicle == ""

        distance_penalty = 2 if (distance < self.__getSafeDistance()) else 0
        distance_penalty += 15 if distance < 2 else 0 #aditional penalty when getting tooooo close to car (before
        # collision)

        target_speed = self.__getTargetSpeed(distance, vehicleId)
        e = target_speed - self._player.getSpeed()
        m_t = 1 if e * e <= 0.25 else 0

        # u = deviation from previous action => encourage low changes in speed
        if _DISCRETE:
            u = self._player.actions[action] - self._player.actions[self.prev_action]
        else:
            u = action - self.prev_action
        # https://nl.mathworks.com/help/reinforcement-learning/ug/train-ddpg-agent
        # -for-adaptive-cruise-control.html
        reward = -(0.1 * e * e + 8 * u * u) + m_t - distance_penalty + self.reward_offset - isCollision * 0
        # speed_reward = -sigmoid(abs(dv/5))*2+2

        # combined_reward = distance_reward*sigmoid(-distance+10)+\
        #                  speed_reward*sigmoid(distance - 10)
        #reward = max(reward, -10)
        return reward

    def __isTerminal(self, distance):
        #check for collision:
        state = self.get_sensor("CollisionSensor").getState()
        if state is None:
            isCollision = False
        else:
            state_frame, state_vehicle = state
            if not state_frame == self.frame:
                isCollision = False
            else:
                isCollision = not state_vehicle == ""


        #is terminal
        if self.evaluation:
            done = self.episodeReward > 509999 or \
                   self.stepCount > 50000 or \
                   isCollision
            # (distance > 1000 and self.agent.getSpeed() >= self.car.getSpeed()) or \
        else:
            done = self.episodeReward > 99999 or \
                   self.stepCount > 5000 or \
                   isCollision
            # (distance > 1000 and self.agent.getSpeed() >= self.car.getSpeed()) or \
        return done

    def __info(self, distance , vehicleId):
        info = {
            'speed':self._player.getSpeed(),
            'set_target_speed':self._player.getTargetSpeed(),
            'optimal_speed': self.__getTargetSpeed(distance, vehicleId),
            'detla V': self._player.getSpeed() - self.__getTargetSpeed(distance, vehicleId),
            'distance': distance,
        }
        return info


    def __getSafeDistance(self):
        return self.t_gap * abs(self.getPlayer().getSpeed()) + self.distance_default

    def __getTargetSpeed(self, distance, leadCarId):
        if (distance == np.Inf or leadCarId == -1):
            return min(self.user_set_point, self.getPlayer().getSpeedLimit())
        save_distance = self.__getSafeDistance()
        #print(f"LeadCar:{leadCarId}")
        lead_car_speed = self.world.get_actor(leadCarId).get_velocity().length()

        # smooth transition of target speed:
        # v1 when far away, v2 when near
        # alpha = 1 when distance > save distance
        # alpha = -1 when distance < save distance
        v1 = min(self.user_set_point, self.getPlayer().getSpeedLimit())
        v2 = min(lead_car_speed, min(self.user_set_point, self.getPlayer().getSpeedLimit()))
        v = self.getPlayer().getSpeed()
        speed = v if not abs(v) < 1e-6 else 1e-5
        speed = max(abs(speed), 4)

        alpha = _sigmoid((distance - save_distance) / abs(speed)) * 2 - 1
        target_speed = alpha * v1 + (1 - alpha) * v2

        #target speed never below zero
        target_speed = max(0, target_speed)
        return target_speed



def main():
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
    args.fps = 20

    config = {
        'device': 'cuda',
        'batch_size': 2048,
        'mini_batch': 4,  # only update once after n experiences
        'num_frames': 1500000,
        'gamma': 0.90,
        'replay_size': 250000,
        'lr': 0.0003,
        'reward_offset': 2.5,
        #epsilon greedy
        'epsilon_start':0.5,
        'epsilon_final': 0.04,
        'epsilon_decay':9000,
        'target_update_freq':2000,
        'history_frames': 3,
        'num_inputs': 12,  # =size of states!
        'num_actions': 101,
        'hidden': [128, 512, 512, 128, 64],
        'debug':False,
    }
    args.agent_config = config

    worldapi=None
    Qlearner.ENABLE_WANDB = True
    try:
        #worldapi = CarlaRLEnv(host="192.168.0.99", config=config, args=args, show = False, debug = False)
        worldapi = CarlaRLEnvFast(config, args)
        #,width=1920, height=1080)
        # fullscreen=True)
        worldapi.spawn(vehicles=50, walkers = 0)
        print(worldapi._player.getTransform().location)

        #config['num_inputs'] = len(worldapi.reset())  # always correct, but expensive in a carla environent
        qlearning = Qlearner(worldapi, DQN, config)
        #qlearning.load("../models/eternal-river-81.pth") # base model
        #qlearning.load("../CarlaEnvironment/models/eager-capybara-144.pth")

        qlearning.train()
        qlearning.save("models/" + qlearning.model_name)
        #qlearning.eval()
        #qlearning.save("../models/TrainedModel_meadow-63_carla.pth")
        #qlearning.save("../models/prime-sun-67_carla.pth")
        worldapi.reset() #force to save episode history
        #while True:
        #    print(worldapi.step(action=8))
    except:
        traceback.print_exc()
    finally:
        if worldapi:
            worldapi.destroy()

    pass

if __name__ == "__main__":
    main()