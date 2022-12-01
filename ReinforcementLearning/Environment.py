import io
import queue
from random import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
from collections import deque
import wandb

__initialDistance = 20
_DISCRETE = True
_MAX_REWARD = 1 # determined myself, used to compute terminal state
                # max episode reward = max_reward / (1-gamma)

_STATE_HISTORY_SIZE = 1

def _sigmoid(x):
    if (x < -512):  # avoid overflow of 'exp' function
        return 0
    return 1 / (1 + math.exp(-x))

def flatten(l: list) ->list:
    return [item for sublist in l for item in sublist]

class DrivingCar:
    def __init__(self, num_actions, speed, position, acceleration):
        self.actions = np.linspace(-1, 1, num_actions)
        self.position = position
        self.speed = speed
        self.acceleration = acceleration

        self.target_speed = 5+random()*15  # 10m/s = 36 km/h ; [5-20]m/s
        self.maxBreaking = 3  # m/s²
        self.maxThrottle = 2.5  # m/s²
        self.dragArea = 0.8
        self.airDensity = 1.204  # kg/m³
        self.mass = 1000  # kg

        self.force = 0.05 + (random()-0.5)/15   #[0,0166, 0.0833]
    def __str__(self) -> str:
        string = "{"
        string += "position: " + str(self.position) + ", "
        string += "speed: " + str(self.speed) + ", "
        string += "acceleration: " + str(self.acceleration) + "}"
        return string

    # def __init__(self, config:dict = None):
    #    config = dict() if config is None else dict()
    #    self.position = config.get("position", 0),
    #    self.speed = config.get("speed", 0),
    #    self.acceleration = config.get("acceleration", 0),

    def getPos(self):
        return self.position

    def getSpeed(self):
        return self.speed

    def setSpeed(self, speed):
        if (self.speed < 0):
            self.speed = 0
        else:
            self.speed = speed

    def updatePos(self, dt=1.0):
        self.position += self.speed * dt

    def updateSpeed(self, dt=1.0):
        self.speed += self.acceleration * dt
        #if self.speed < 0:
        #    self.speed = 0

    def setAcceleration(self, acceleration):
        if acceleration > self.maxThrottle:
            self.acceleration = self.maxThrottle
        elif acceleration < (-self.maxBreaking):
            self.acceleration = -self.maxBreaking
        else:
            self.acceleration = acceleration

    def updateAcceleration(self, action=None, dt=1.0):
        # compute force from action
        if action is None:
            force = self.force
        else:
            if _DISCRETE:
                force = self.actions[action] # descrete action space
            else:
                force = action  # continues action space
            if (force < 0):
                # break force
                force *= self.maxBreaking
            else:
                # throttle
                force *= self.maxThrottle

        # compute force from drag
        drag = self.dragArea * self.speed * self.airDensity / (2 * self.mass) * self.speed
        if (self.speed < 0):
            drag = -drag

        self.acceleration = force - drag

    def printStats(self):
        print("speed: ", self.speed)
        print("position: ", self.position)
        print("acceleration: ", self.acceleration)

    def step(self, action=None, dt=1.0):
        self.updateAcceleration(action, dt)
        self.updateSpeed(dt)
        self.updatePos(dt)

    def reset(self):
        self.position = 0
        self.speed = 0
        self.acceleration = 0

    def type(self):
        return "DrivingCar"

    def plot(self, stepCount,dt=1.0):
        pos=[0.0]
        speed=[0.0]
        acc=[0.0]
        self.reset()
        for s in range(stepCount):
            self.step(None, dt)
            pos.append(self.getPos())
            speed.append(self.getSpeed())
            acc.append((speed[-1]-speed[-2])/dt)
        pos= pos[2:]
        speed= speed[2:]
        acc= acc[2:]
        x = np.linspace(0, stepCount-2, num=stepCount-1)
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        plt.suptitle(self.type())
        ax1.plot(x, pos)
        ax1.set_title("position")
        ax2.plot(x, speed)
        ax2.set_title("speed")
        ax3.plot(x, acc)
        ax3.set_title("acceleration")
        plt.show()


class SinusCar(DrivingCar):

    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, speed, position, 0)
        self.frame = 0
        self.amplitude = 3 + (random()-0.5)*2 #[2,4]
        self.period = 200 + (random()-0.5)*40 #[180,220]
        self.offset = 3 + random()*3          #[3,6]

    def step(self, action=None, dt=1.0):
        self.setSpeed(self.amplitude * math.sin(self.frame / self.period) + self.offset)
        self.updatePos(dt)
        self.frame += 1

    def type(self):
        return "Sinuscar"


class ConstantCar(DrivingCar):

    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, speed, position, 0)
        self.frame = 0

    def step(self, action=None, dt=1.0):
        self.updatePos(dt)
        self.frame += 1

    def type(self):
        return "ConstantCar"


class NoDrivingCar(DrivingCar):

    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, 0, position, 0)
        self.frame = 0

    def step(self, action=None, dt=1.0):
        self.frame += 1

    def type(self):
        return "NoDrivingCar"


class SpeedUpBrakeCar(DrivingCar):

    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, speed, position, acceleration)
        self.frame = 0
        self.brakeCounter = 0

    def step(self, action=None, dt=1.0):
        if self.speed <= 0:
            self.setAcceleration(np.random.rand()*self.maxThrottle)  # random acceleration
            self.updateSpeed(dt)
            self.brakeCounter = 0
        else:
            if self.speed < self.target_speed:
                self.updateSpeed(dt)
            else:
                if self.brakeCounter < 100:
                    self.speed = self.target_speed
                    self.brakeCounter += 1
                else:
                    self.setAcceleration(-self.maxBreaking*0.8)
                    self.updateSpeed(dt)
                    self.brakeCounter += 1
        self.updatePos(dt)
        self.frame += 1

    def type(self):
        return "SpeedUpBrakeCar"

class RandomCar(DrivingCar):
    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, speed, position, acceleration)
        self.frame = 0
        self.counter = 0
        self.constant = np.random.randint(0, 100)

    def step(self, action=None, dt=1.0):
        if self.counter < self.constant:
            self.counter += 1
        else:
            if self.speed <= 0:
                self.setAcceleration(np.random.randint(1, self.maxThrottle) * (np.random.random()))
            else:
                if self.speed >= self.target_speed:
                    self.setAcceleration(np.random.randint(1, self.maxThrottle) * (-1 + np.random.random()))
                else:
                    self.setAcceleration(-1 + 2 * np.random.random())
            self.counter = 0
            self.constant = np.random.randint(0, 100)
        self.updateSpeed(dt)
        self.updatePos(dt)
        self.frame += 1

    def type(self):
        return "RandomCar"

class BackDrivingCar(DrivingCar):

    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, -speed, position, 0)
        self.frame = 0

    def step(self, action=None, dt=1.0):
        self.updatePos(dt)
        self.frame += 1

    def type(self):
        return "BackDrivingCar"


class SimpleACC(gym.Env):
    def __init__(self, config: dict):
        self.setpoint = 20
        self.reward_offset = config.get('reward_offset',1.5)
        if _DISCRETE:

            #self.action_space = gym.spaces.Box(low=np.array([0]),high=np.array([config.get("num_actions", 7)]),
            #                                   dtype=int)
            self.action_space = gym.spaces.Discrete(7)
            low = np.tile(np.array([0, -np.inf, -np.inf, 0]), config.get('history_frames', 3)).flatten()
            high = np.tile(np.array([np.inf, np.inf, np.inf, 7]), config.get('history_frames', 3)).flatten()
            self.observation_space = gym.spaces.Box(low=low, high=high)
        else:
            self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]))
            low = np.tile(np.array([0, -np.inf, -np.inf, -1]), config.get('history_frames', 3)).flatten()
            high = np.tile(np.array([np.inf, np.inf, np.inf, 1]), config.get('history_frames', 3)).flatten()
            self.observation_space = gym.spaces.Box(low=low, high=high)
        #spaces = {
        # distance, self.agent.target_speed, self.agent.getSpeed(), self.prev_action
        #}
        self.frames = deque(maxlen=config.get('history_frames', 3))
        # distance,distance prev,  speedAgent, targetspeed, prev_action

        self.config = config
        self.agent_config = config.get("agent", dict())
        self.car_config = config.get("car", dict())
        self.agent = DrivingCar(7, 0, 0, 2)  # num_actions, startspeed, startposition, starting acceleration
        self.car = self.chooseRandomCar(7, 20, 200, 2)

        self.stepCount = 0
        self.history = {'agent': [], 'car': [], 'agent_speed': [], 'car_speed': [], 'target_speed':[], 'safe_dist':[],
                        'reward':[]}


        # https://nl.mathworks.com/help/reinforcement-learning/ug/train-ddpg-agent-for-adaptive-cruise-control.html
        self.distance_default = 4  # m
        self.t_gap = 2

        self.episodeReward = 0.0
        self.gamma = config.get('gamma',0.99)
        self.prev_action = 0

        self.train() #set training mode

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        self.env_plot = fig
        self.env_plot_axis = (ax1, ax2, ax3)


    def __str__(self) -> str:
        string = ""
        string += "Agent:\t" + str(self.agent) + "\n"
        string += "Car  :\t" + str(self.car) + "\n"
        string += "State:\t" + str(self.__getState()) + "\n"
        return string

    def chooseRandomCar(self, num_actions, speed, position, acceleration):
        number_of_cars = 6
        choice = np.random.randint(0, number_of_cars+1)

        if choice == 0:
            return DrivingCar(num_actions,speed, position, acceleration)
        elif choice == 1:
            return SinusCar(num_actions, speed, position, acceleration)
        elif choice == 2:
            return ConstantCar(num_actions, speed, position, acceleration)
        elif choice == 3:
            return NoDrivingCar(num_actions, speed, position, acceleration)
        elif choice == 4:
            return SpeedUpBrakeCar(num_actions, speed, position, acceleration)
        elif choice == 5:
            return RandomCar(num_actions, speed, position, acceleration)
        elif choice == 6:
            return BackDrivingCar(num_actions, speed, position, acceleration)

    def step(self, action):
        # step state
        dt = 0.05
        self.car.step(dt=dt)
        self.agent.step(action=action, dt=dt)
        self.stepCount += 1

        self.frames.append(self.__getState())
        state = np.array(list(self.frames)).flatten()

        # reward
        reward = self.__getReward(action)

        # done
        done = self.__isTerminal()

        # info
        info = self.__info()

        self.history['agent'].append(self.agent.getPos())
        self.history['car'].append(self.car.getPos())
        self.history['safe_dist'].append(self.__getSafeDistance())
        self.history['agent_speed'].append(self.agent.getSpeed())
        self.history['car_speed'].append(self.car.getSpeed())
        self.history['target_speed'].append(self.__getTargetSpeed(distance=self.car.getPos() - self.agent.getPos()))
        self.history['reward'].append(reward)

        self.episodeReward += reward
        self.prev_action = action
        return state, reward, done, info

    def reset(self, **kwargs):
        self.agent = DrivingCar(
            num_actions=self.config.get("num_actions", 7),
            position=self.agent_config.get("position", 0),
            speed=self.agent_config.get("speed", 8+random()*12),
            acceleration=self.agent_config.get("acceleration", 2))

        position = self.agent_config.get("position", 200)
        speed = self.agent_config.get("speed", 5)
        acceleration = self.agent_config.get("acceleration", 0)
        position += (random() - 0.5) * 200  # ±100m
        speed += (random() - 0.25) * 20  # initial speed += [-5,15]
        acceleration += random()
        self.car = self.chooseRandomCar(
            num_actions=self.config.get("num_actions", 7),
            position=position,
            speed=speed,
            acceleration=acceleration)

        self.stepCount = 0
        self.history = {'agent': [], 'car': [], 'agent_speed': [], 'car_speed': [], 'target_speed':[], 'safe_dist':[],
                        'reward':[]}
        self.prev_action = 0
        self.episodeReward = 0

        self.frames = deque(maxlen=self.config.get('history_frames', 3))
        while len(self.frames) < self.config.get('history_frames', 3):
            self.frames.append(self.__getState())

        return np.array(list(self.frames)).flatten()

    def __exit__(self, *args):
        return super().__exit__(*args)

    def __getState(self):
        """
        Compute the state of the current environment
        :return:
        """
        distance = self.car.getPos() - self.agent.getPos()
        distance = np.clip(distance, -500, 500)
        #if (len(self.history['car']) > 1):
        #prevDistance = self.history['car'][-2] - self.history['agent'][-2]
        #prevSpeed = self.history['agent_speed'][-2]
        #else:
        #prevDistance = distance
        #prevSpeed = self.agent.getSpeed()

        #prevDistance = np.clip(prevDistance,-500,500)

        return distance, self.agent.target_speed, self.agent.getSpeed(), self.prev_action

    def __getReward(self,action):
        """
        Compute the state of the last step
        """
        # TODO: think about the reward function
        distance = abs(self.car.getPos() - self.agent.getPos())

        distance_penalty = 2 if (distance < self.__getSafeDistance()) else 0


        target_speed = self.__getTargetSpeed(distance)
        e = target_speed - self.agent.getSpeed()
        m_t = 1 if e * e <= 0.25 else 0

        #u = deviation from previous action => encourage low changes in speed
        if _DISCRETE:
            u = self.agent.actions[action] - self.agent.actions[self.prev_action]
        else:
            u = action-self.prev_action  ## TODO: what is U????
        # https://nl.mathworks.com/help/reinforcement-learning/ug/train-ddpg-agent
        # -for-adaptive-cruise-control.html
        reward = -(0.1 * e * e + u * u) + m_t - distance_penalty  + self.reward_offset
        # speed_reward = -sigmoid(abs(dv/5))*2+2

        # combined_reward = distance_reward*sigmoid(-distance+10)+\
        #                  speed_reward*sigmoid(distance - 10)
        reward = max(reward,-10)
        return reward

    def __getTargetSpeed(self, distance):
        save_distance = self.__getSafeDistance()
        #if distance > save_distance:
        #    target_speed = self.agent.target_speed
        #else:
        #    target_speed = min(self.car.getSpeed(),self.agent.target_speed)

        #smooth transition of target speed:
        #v1 when far away, v2 when near
        # alpha = 1 when distance > save distance
        # alpha = -1 when distance < save distance
        v1 = self.agent.target_speed
        v2 = min(self.car.getSpeed(), self.agent.target_speed)
        speed = self.agent.getSpeed() if not abs(self.agent.getSpeed()) < 1e-6 else 1e-5
        speed = max(abs(speed),4)

        alpha = _sigmoid((distance - save_distance)/abs(speed)) * 2 - 1
        target_speed = alpha * v1 + (1 - alpha) * v2


        return target_speed

    def __getSafeDistance(self):
        return self.t_gap * abs(self.agent.getSpeed()) + self.distance_default

    def __isTerminal(self):
        """
        Return true when a terminal state is reached
        """
        distance = self.car.getPos() - self.agent.getPos()
        if self.evaluation:
            done = distance <= 0 or \
                   self.episodeReward > 50999 or \
                   self.stepCount > 50000
        else:
            done = distance <= 0 or \
                   self.episodeReward > 9999 or \
                   self.stepCount > 5000
        return done

    def __info(self):
        info = {}
        if self.stepCount > 5000:
            info['TimeLimit.truncated'] = True
        distance = self.car.getPos() - self.agent.getPos()
        if distance <= 0.1:
            info['collision'] = True
        return info

    def printCars(self):
        print("Agent:")
        self.agent.printStats()
        print("Car:")
        self.car.printStats()

    def plot(self):
        if (len(self.history['agent']) <= 1):
            print("Empty history")
            return
        distance = []
        for ego, agent in zip(self.history['agent'], self.history['car']):
            distance.append(agent - ego)



        x = np.linspace(0, self.stepCount - 1, num=self.stepCount)
        #self.env_plot_axis[0].cla()
        #self.env_plot_axis[1].cla()
        #self.env_plot_axis[2].cla()
        #self.env_plot_axis[0].plot(x, self.history['agent'], color='blue')
        #self.env_plot_axis[0].plot(x, self.history['car'], color='orange')
        #self.env_plot_axis[0].set_title('position')
        #self.env_plot_axis[1].plot(x, self.history['agent_speed'], color='blue')
        #self.env_plot_axis[1].plot(x, self.history['car_speed'], color='orange')
        #self.env_plot_axis[1].plot(x, self.history['target_speed'], color='green')
        #self.env_plot_axis[1].set_title('speed')
        #self.env_plot_axis[2].plot(x, distance, color='blue')
        #self.env_plot_axis[2].plot(x, self.history['safe_dist'], color='green')
        #self.env_plot_axis[2].set_title('distance')
        #self.env_plot.canvas.draw()
        #self.env_plot.canvas.flush_events()
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
        ax1.plot(x, self.history['agent'], color='blue')
        ax1.plot(x, self.history['car'], color='orange')
        ax1.set_title('position')
        ax2.plot(x, self.history['agent_speed'], color='blue')
        ax2.plot(x, self.history['car_speed'], color='orange')
        ax2.plot(x, self.history['target_speed'], color='green')
        ax2.set_title('speed')
        ax3.plot(x, distance, color='blue')
        ax3.plot(x, self.history['safe_dist'], color='green')
        ax3.set_title('distance')

        ax4.set_title('reward')
        ax4.plot(x, self.history['reward'])

        plt.show(block = False)


        #wandb.log({'agent_pos': self.history['agent'],
        #           'car_pos': self.history['car'],
        #           'agent_vel': self.history['agent_speed'],
        #           'car_vel': self.history['car_speed'],
        #           'distance': distance
        #           })

    def eval(self):
        #set environment in evaluation mode:
        self.evaluation = True
    def train(self):
        self.evaluation = False
    def render(self, mode="human"):
        pass

    def getTypeOfLeadingCar(self):
        type = self.car.type()
        if type =='DrivingCar':
            return 0
        if type =='Sinuscar':
            return 1
        if type =='ConstantCar':
            return 2
        if type =='NoDrivingCar':
            return 3
        if type =='SpeedUpBrakeCar':
            return 4
        if type =='RandomCar':
            return 5


if __name__ == "__main__":
    #plot all different cars:
    drivingCar = DrivingCar(1,0,0,0)
    sinusCar = SinusCar(1,0,0,0)
    constantCar = ConstantCar(1,0,0,0)
    noDrivingCar = NoDrivingCar(1,0,0,0)
    speedUpBrakeCar = SpeedUpBrakeCar(1,0,0,0)
    randomCar = RandomCar(1,0,0,0)

    drivingCar.plot(4000,0.05)
    sinusCar.plot(4000,0.05)
    constantCar.plot(4000,0.05)
    noDrivingCar.plot(4000,0.05)
    speedUpBrakeCar.plot(4000,0.05)
    randomCar.plot(4000,0.05)