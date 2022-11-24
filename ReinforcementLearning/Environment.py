import numpy as np
import matplotlib.pyplot as plt
import math

import wandb

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
        self.maxBreaking = 3  # m/s²
        self.maxThrottle = 2  # m/s²
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

    def setSpeed(self, speed):
        if (self.speed < 0 ):
            self.speed = 0
        else:
            self.speed = speed

    def updatePos(self, dt=1.0):
        self.position += self.speed * dt

    def updateSpeed(self, dt=1.0):
        self.speed += self.acceleration * dt
        if self.speed < 0 :
            self.speed = 0

    def setAcceleration(self,acceleration):
        if acceleration>self.maxThrottle:
            self.acceleration=self.maxThrottle
        elif acceleration<(-self.maxBreaking):
            self.acceleration = -self.maxBreaking
        else:
            self.acceleration = acceleration


    def updateAcceleration(self, action=None, dt=1.0):
        # compute force from action
        if action is None:
            force = 0.05
        else:
            force = action # continues action space
            #force = self.actions[action] # descrete action space
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
        self.updateSpeed()
        self.updatePos()

class SinusCar(DrivingCar):

    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, speed, position, 0)
        self.frame = 0

    def step(self, action=None, dt=1.0):
        self.setSpeed(2*math.sin(self.frame/200)+3)
        self.updatePos()
        self.frame += 1


class ConstantCar(DrivingCar):

    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, speed, position, 0)
        self.frame = 0

    def step(self, action=None, dt=1.0):
        self.updatePos()
        self.frame += 1

class NoDrivingCar(DrivingCar):

    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, 0, position, 0)
        self.frame = 0

    def step(self, action=None, dt=1.0):
        self.frame += 1

class SpeedUpBrakeCar(DrivingCar):

    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, speed, position, acceleration)
        self.frame = 0
        self.brakeCounter = 0

    def step(self, action=None, dt=1.0):
        if self.speed<=0:
            self.setAcceleration(np.random.randint(1,self.maxThrottle)) #random acceleration
            self.updateSpeed()
            self.brakeCounter=0
        else:
            if self.speed<83:
                self.updateSpeed()
            else:
                if self.brakeCounter<100:
                    self.speed = 83
                    self.brakeCounter += 1
                else:
                    self.setAcceleration(-self.maxBreaking)
                    self.updateSpeed()
                    self.brakeCounter +=1
        self.updatePos()
        self.frame += 1

class RandomCar(DrivingCar):
    def __init__(self, num_actions, speed, position, acceleration):
        super().__init__(num_actions, speed, position, acceleration)
        self.frame = 0
        self.counter=0
        self.constant=np.random.randint(0,100)

    def step(self, action=None, dt=1.0):
        if self.counter<self.constant:
            self.counter +=1
        else:
            if self.speed<=0:
                self.setAcceleration(np.random.randint(1,self.maxThrottle)*(np.random.random()))
            else:
                if self.speed>=80:
                    self.setAcceleration(np.random.randint(1,self.maxThrottle)*(-1+np.random.random()))
                else:
                    self.setAcceleration(-1+2*np.random.random())
            self.counter=0
            self.constant = np.random.randint(0, 100)
        self.updateSpeed()
        self.updatePos()
        self.frame += 1

class Environment:
    def __init__(self, config: dict):
        self.config = config
        self.agent_config = config.get("agent", dict())
        self.car_config = config.get("car", dict())
        self.agent = DrivingCar(1, 0, 0, 2) #num_actions, startspeed, startposition, starting acceleration
        self.car = self.chooseRandomCar(1,20,200,2)

        self.stepCount = 0
        self.history = {'agent': [], 'car': [], 'agent_speed':[], 'car_speed':[]}

        #https://nl.mathworks.com/help/reinforcement-learning/ug/train-ddpg-agent-for-adaptive-cruise-control.html
        self.distance_default = 4  # m
        self.t_gap = 1.4

    def __str__(self) -> str:
        string = ""
        string += "Agent:\t" + str(self.agent) + "\n"
        string += "Car  :\t" + str(self.car) + "\n"
        string += "State:\t" + str(self.__getState()) + "\n"
        return string

    def chooseRandomCar(self,num_actions,speed,position,acceleration):
        choice=np.random.randint(0,5)
        if choice==0:
            return ConstantCar(num_actions,speed,position,acceleration)
        elif choice==1:
            return SinusCar(num_actions,speed,position,acceleration)
        elif choice==2:
            return RandomCar(num_actions,speed,position,acceleration)
        elif choice==3:
            return SpeedUpBrakeCar(num_actions,speed,position,acceleration)
        elif choice==4:
            return NoDrivingCar(num_actions,speed,position,acceleration)

    def step(self, action):
        self.car.step(dt=0.1)
        self.agent.step(action=action, dt=0.1)
        self.stepCount += 1

        self.history['agent'].append(self.agent.getPos())
        self.history['car'].append(self.car.getPos())
        self.history['agent_speed'].append(self.agent.getSpeed())
        self.history['car_speed'].append(self.car.getSpeed())

        return self.__getState(), self.__getReward(), self.__isTerminal(), self.__info()

    def reset(self):
        self.agent = DrivingCar(
            num_actions=self.config.get("num_actions", 1),
            position=self.agent_config.get("position", 0),
            speed=self.agent_config.get("speed", 0),
            acceleration=self.agent_config.get("acceleration", 2))
        self.car = self.chooseRandomCar(
            num_actions=self.config.get("num_actions", 1),
            position=self.agent_config.get("position", 200),
            speed=self.agent_config.get("speed", 5),
            acceleration=self.agent_config.get("acceleration", 0))

        self.stepCount = 0
        self.history = {'agent': [], 'car': [], 'agent_speed': [], 'car_speed': []}

        return self.__getState()

    def __getState(self):
        """
        Compute the state of the current environment
        :return:
        """
        distance = self.car.getPos() - self.agent.getPos()
        if (len(self.history['car']) >1):
            prevDistance = self.history['car'][-2] - self.history['agent'][-2]
        else:
            prevDistance = distance
        distance = np.clip(distance,-200, 200)
        return distance, prevDistance ,self.agent.getSpeed(), self.agent.target_speed

    def __getReward(self):
        """
        Compute the state of the last step
        """
        #TODO: think about the reward function
        distance = self.car.getPos() - self.agent.getPos()
        #distance_reward = sigmoid(-(distance-8)/3)  # closer = better

        save_distance = self.t_gap * self.agent.getSpeed()+self.distance_default
        if distance > save_distance:
            target_speed = self.agent.target_speed
        else:
            target_speed = self.car.getSpeed()
        e = target_speed - self.agent.getSpeed()
        m_t = 1 if e*e <= 0.25 else 0
        u = 0 ## TODO: what is U???? https://nl.mathworks.com/help/reinforcement-learning/ug/train-ddpg-agent-for-adaptive-cruise-control.html
        reward = -(0.1 * e*e + u * u) + m_t
        #speed_reward = -sigmoid(abs(dv/5))*2+2

        #combined_reward = distance_reward*sigmoid(-distance+10)+\
        #                  speed_reward*sigmoid(distance - 10)

        return reward

    def __isTerminal(self):
        """
        Return true when a terminal state is reached
        """
        distance = self.car.getPos() - self.agent.getPos()
        return self.agent.position > 2000 or distance < 4 or self.stepCount > 1000

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
        if (len(self.history['agent']) <= 1):
            print("Empty history")
            return
        distance = []
        for ego,agent in zip(self.history['agent'],self.history['car']):
            distance.append(agent-ego)

        fig, (ax1, ax2, ax3) = plt.subplots(1,3)

        x = np.linspace(0, self.stepCount - 1, num=self.stepCount)
        ax1.plot(x, self.history['agent'])
        ax1.plot(x, self.history['car'])
        ax1.set_title('position')
        ax2.plot(x, self.history['agent_speed'])
        ax2.plot(x, self.history['car_speed'])
        ax2.set_title('speed')
        ax3.plot(x,distance)
        ax3.set_title('distance')
        plt.show()

        wandb.log({'agent_pos': self.history['agent'],
                   'car_pos':self.history['car'],
                   'agent_vel': self.history['agent_speed'],
                   'car_vel': self.history['car_speed'],
                   'distance': distance
                    })
