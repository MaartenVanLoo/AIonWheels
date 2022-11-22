import numpy as np
import matplotlib.pyplot as plt
import math

__initialDistance = 20


def sigmoid(x):
    if (x < -512): #avoid overflow of 'exp' function
        return 0
    return 1 / (1 + math.exp(-x))


class DrivingCar:
    def __init__(self, num_actions, speed, position, acceleration):
        self.actions = np.linspace(-1, 1, num_actions)
        self.position = position
        self.speed = speed
        self.acceleration = acceleration

        self.target_speed = 20  # 20m/s = 72 km/h
        self.maxBreaking = 3.5  # m/s²
        self.maxThrottle = 5.5  # m/s²
        self.dragArea = 0.8
        self.airDensity = 1.204  # kg/m³
        self.mass = 1000  # kg

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

    def updatePos(self, dt):
        self.position += self.speed * dt

    def updateSpeed(self, dt):
        self.speed += self.acceleration * dt

    def updateAcceleration(self, action=None, dt=1):
        # compute force from action
        if action is None:
            force = 0.05
        else:
            force = self.actions[action]
            if (force < 0):
                # break force
                force *= self.maxBreaking
            else:
                # throttle
                force *= self.maxThrottle

        # compute force from drag
        drag = self.dragArea * self.speed * self.airDensity / (2 * self.mass) * self.speed
        self.acceleration = force - drag

    def printStats(self):
        print("speed: ", self.speed / 3.6)
        print("position: ", self.position)
        print("acceleration: ", self.acceleration)

    def step(self, action=None, dt=1):
        self.updateAcceleration(action, dt)
        self.updateSpeed(dt)
        self.updatePos(dt)


class Environment:
    def __init__(self, config: dict):
        self.config = config;
        self.agent_config = config.get("agent", dict())
        self.car_config = config.get("car", dict())
        self.agent = DrivingCar(7, 0, 0, 45)
        self.car = DrivingCar(7, 50, 200, 0)

        self.stepCount = 0;
        self.history = {'agent': [], 'car': []};

    def __str__(self) -> str:
        string = ""
        string += "Agent:\t" + str(self.agent) + "\n"
        string += "Car  :\t" + str(self.car) + "\n"
        string += "State:\t" + str(self.__getState()) + "\n"
        return string

    def step(self, action):
        self.car.step(dt=0.05)
        self.agent.step(action=action, dt=0.05)
        self.stepCount += 1

        self.history['agent'].append(self.agent.getPos())
        self.history['car'].append(self.car.getPos())

        return self.__getState(), self.__getReward(), self.__isTerminal(), self.__info()

    def reset(self):
        self.agent = DrivingCar(
            num_actions=self.config.get("num_actions", 7),
            position=self.agent_config.get("position", 0),
            speed=self.agent_config.get("speed", 0),
            acceleration=self.agent_config.get("acceleration", 0))
        self.car = DrivingCar(
            num_actions=self.config.get("num_actions", 7),
            position=self.agent_config.get("position", 20),
            speed=self.agent_config.get("speed", 5),
            acceleration=self.agent_config.get("acceleration", 0))

        self.stepCount = 0
        self.history = {'agent': [], 'car': []}

        return self.__getState()

    def __getState(self):
        """
        Compute the state of the current environment
        :return:
        """
        distance = self.car.getPos() - self.agent.getPos()
        return distance, self.agent.getSpeed()

    def __getReward(self):
        """
        Compute the state of the last step
        """
        #TODO: think about the reward function
        distance = self.car.getPos() - self.agent.getPos()
        reward = 1 / math.sqrt(distance);  # closer = better

        dv = self.agent.target_speed - self.agent.getSpeed()  # slower = higher dv
        reward += sigmoid(-dv)-.5
        return reward

    def __isTerminal(self):
        """
        Return true when a terminal state is reached
        """
        distance = self.car.getPos() - self.agent.getPos()
        return self.agent.position > 2000 or distance < 4 or self.stepCount > 4000

    def __info(self):
        info = dict()
        if self.stepCount > 4000:
            info['TimeLimit.truncated'] = True
        return info

    def printCars(self):
        print("Agent:")
        self.agent.printStats()
        print("Car:")
        self.car.printStats()

    def plot(self):
        x = np.linspace(0, self.stepCount - 1, num=self.stepCount)
        plt.plot(x, self.history['agent'])
        plt.plot(x, self.history['car'])
        plt.show()
