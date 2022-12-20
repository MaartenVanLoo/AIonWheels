from __future__ import print_function

import glob
import logging
import os
import numpy.random as random
import sys

try:
    import pygame
    from collections import deque
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

from CarlaAgents import CarlaAgentRL  # pylint: disable=import-error
from basic_agent import BasicAgent

import CarlaImport


class CarlaConnection:
    def __init__(self, config) -> None:
        super().__init__()

        # carla argument constants
        self.debug = True
        self.host = "127.0.0.1"
        self.port = 2000
        self.height = 1280
        self.width = 720
        self.filter = "vehicle.*"
        self.seed = None
        self.sync = True

        # carla world objects
        self.clock = None
        self.controller = None
        self.world = None
        self.display = None
        self.traffic_manager = None

        # follower agent, initially inactive
        self.agent = None
        # leading car, will be initialized in init function
        self.leading_car = None
        # dequeue for cars destination
        self.locations_buffer = deque()
        self.clock_ticks = 0
        # first destination, spawn for agent
        self.initial_destination = None
        # random spawn points, initialized in init function
        self.spawn_points = None
        # initial blueprint to spawn actor
        self.blueprint = None
        # initial transform, where leading_car is spawned
        self.initial_transform = None

        self.num_actions = config.get('num_actions', 7)

        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
        logging.info('listening to server %s:%s', self.host, self.port)

    def init(self):
        """
        Init function, starts carla environment
         also initializes necesary object variables
        :return: None
        """
        pygame.init()
        pygame.font.init()

        if self.seed:
            random.seed(self.seed)

        client = carla.Client(self.host, self.port)
        client.set_timeout(4.0)

        self.traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if self.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            self.traffic_manager.set_synchronous_mode(True)

        self.display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        # carla python api code
        hud = CarlaImport.HUD(self.width, self.height)
        self.world = CarlaImport.World(client.get_world(), hud, self.sync, self.filter)
        self.spawn_points = self.world.map.get_spawn_points()
        self.controller = CarlaImport.KeyboardControl(self.world)

        # spawn leading_car car and keep starting point
        self.initial_transform = self.world.player.get_transform()
        self.leading_car = BasicAgent(self.world.player)

        # Set first destination
        spawn_points = self.world.map.get_spawn_points()
        self.initial_destination = random.choice(spawn_points).location
        self.locations_buffer.append(self.initial_destination)
        self.leading_car.set_destination(self.initial_destination)

        self.clock = pygame.time.Clock()

        # loop till agent is spawned
        self.initial_loop()

    def step(self, action: int):
        """
        Compute action (acceleration) TODO
        Compute a step in the carla environment
        observe environment (compute state) TODO
        :return state, reward, done, info TODO
        """
        self.clock.tick()
        if self.sync:
            self.world.world.tick()
        else:
            self.world.world.wait_for_tick()
        if self.controller.parse_events():
            return

        self.world.tick(self.clock)
        self.world.render(self.display)
        pygame.display.flip()

        # Rerouting to new location for leading_car car
        if self.leading_car.done():
            new_dest = random.choice(self.spawn_points).location
            self.locations_buffer.append(new_dest)
            self.leading_car.set_destination(new_dest)
            print("[!] Leading car: The target has been reached, searching for another target")

        # Control of leading_car car
        control = self.leading_car.run_step()
        control.manual_gear_shift = False
        self.leading_car._vehicle.apply_control(control)

        # after 100 iters of 0.05 seconds(5 seconds) initialize follower car
        if self.agent is None:
            self.clock_ticks += 1
            if self.clock_ticks == 100:
                self.world.restart(self.initial_transform)
                self.agent = CarlaAgentRL(self.world.player, self.num_actions)
                self.agent.set_destination(self.initial_destination)
                print(self.get_distance())
            return

        # Rerouting to new location for follower car
        if self.agent.done():
            self.agent.set_destination(self.locations_buffer.popleft())
            print("[!] Agent_controlled: The target has been reached, searching for another target")

        # Control of follower car
        control_model = self.agent.run_step(action)
        control_model.manual_gear_shift = False
        self.world.player.apply_control(control_model)
        return self.get_state(), self.get_reward(), self.is_terminal(), {}

    def initial_loop(self):
        """
        Initial loop so that agent is spawned
        :return: None
        """
        try:
            for i in range(100):
                _ = self.step(0)
        except Exception as e:
            print(e)

    def cleanup(self):
        """
        Clean up python carla environment
        call always when ending
        :return: None
        """
        if self.world is not None:
            settings = self.world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(True)
            self.world.destroy()

        if self.leading_car is not None:
            self.leading_car._vehicle.destroy()

        pygame.quit()

    def get_reward(self):
        """
        Will get the reward for the step TODO
        :return: None
        """
        pass

    def is_terminal(self):
        """
        Will compute if it is terminal state or not TODO
        :return: bool
        """
        pass

    def restart(self):
        """
        Restart full carla environment
        :return: None
        """
        self.cleanup()
        self.init()

    def get_speed_agent(self):
        """
        Get the speed of the agent car
        :return: float
        """
        return self.agent.get_speed()

    def get_speed_leading_car(self):
        """
        Get the speed of the leading car
        :return: float
        """
        self.leading_car.get_speed()

    def get_distance(self):
        """
        Get the distance between leading car and agent
        :return: float
        """
        return self.get_agent_location().distance(self.get_leading_car_location())

    def get_agent_location(self):
        """
        Get the location of the front of the agent car
        :return: carla.Location(3D vector)
        """
        location = self.agent._vehicle.bounding_box.location
        location.y = location.y + self.agent._vehicle.bounding_box.extent.y
        return location

    def get_leading_car_location(self):
        """
        Get the location of the back of the leading car
        :return: carla.Location(3D vector)
        """
        location = self.leading_car._vehicle.bounding_box.location
        location.y = location.y - self.leading_car._vehicle.bounding_box.extent.y
        return location

    # TODO
    def get_state(self):
        """
        Gets the current state of environment
        :return:
        """
        pass

    def plot(self):
        pass

    def reset(self):
        """
        Restarts the environment and returns first state
        :return: returns a state
        """
        self.restart()
        return self.step(0)
