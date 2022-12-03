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
from carla import ColorConverter as cc

from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

import CarlaImport

class CarlaConnection:
    def __init__(self) -> None:
        super().__init__()

        # carla argument constants
        self.debug = True
        self.host = "127.0.0.1"
        self.port = 2000
        self.height = 1280
        self.width = 720
        self.filter = "vehicle.*"
        self.seed = None
        self.sync = False

        # carla world objects
        self.clock = None
        self.controller = None
        self.world = None
        self.display = None
        self.traffic_manager = None 
    

        # follower agent, initially inactive
        self.agent = None
        self.leading_car = None
        self.locations_buffer = deque()
        self.clock_ticks = 0
        self.initial_destination = None

    def main(self):
        """Main method"""

        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
        logging.info('listening to server %s:%s', self.host, self.port)
        
        print(__doc__)

        try:
            self.init()
            self.game_loop()

        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
        except Exception:
            if self.world is not None:
                if self.leading_car is not None:
                    self.leading_car.destroy()
                self.world.destroy()
            
    def init(self):
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

        hud = CarlaImport.HUD(self.width, self.height)
        self.world = CarlaImport.World(client.get_world(), hud, self.sync, self.filter)
        self.controller = CarlaImport.KeyboardControl(self.world)

        '''
        transform = world.player.get_transform()
        transform_fv = transform.get_forward_vector()
        transform_fv.x, transform_fv.y, transform_fv.z = normalize3Dvector(transform_fv.x, transform_fv.y, transform_fv.z)
        new_location = transform.location + (transform_fv * -50)
        transform.location = new_location
        '''

        # spawn leading_car car
        blueprint = random.choice(self.world.world.get_blueprint_library().filter('vehicle.*.*'))
        initial_transform = self.world.player.get_transform()
        vehicle_bp = self.world.world.spawn_actor(blueprint, initial_transform)
        agent = BasicAgent(vehicle_bp)

        # Set first destination
        spawn_points = self.world.map.get_spawn_points()
        self.initial_destination = random.choice(spawn_points).location
        self.locations_buffer.append(self.initial_destination)
        agent.set_destination(self.initial_destination)

        self.clock = pygame.time.Clock()

    def game_loop(self):
        """
        Main loop of the simulation. It handles updating all the HUD information,
        ticking the agent and, if needed, the world.
        """
        spawn_points = self.world.map.get_spawn_points()
        try:
            while True:
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
                    new_dest = random.choice(spawn_points).location
                    self.locations_buffer.append(new_dest)
                    self.leading_car.set_destination(new_dest)
                    print("Agent target: The target has been reached, searching for another target")

                # Control of leading_car car
                control = self.leading_car.run_step()
                control.manual_gear_shift = False
                self.leading_car.vehicle.apply_control(control)

                # after 100 iters of 0.05 seconds(5 seconds) initialize follower car
                if self.agent is None:
                    self.clock_ticks += 1
                    if self.clock_ticks == 100:
                        self.agent = BasicAgent(self.world.player)
                        self.agent.set_destination(self.initial_destination)
                        print(self.get_distance())
                    continue

                # Rerouting to new location for follower car
                if self.agent.done():
                    self.agent.set_destination(self.locations_buffer.popleft())
                    print("Agent_controlled: The target has been reached, searching for another target")

                # Control of follower car
                control_model = self.agent.run_step()
                control_model.manual_gear_shift = False
                self.world.player.apply_control(control_model)

        finally:

            if self.world is not None:
                settings = self.world.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.world.apply_settings(settings)
                self.traffic_manager.set_synchronous_mode(True)

                self.world.destroy()
                if self.leading_car is not None:
                    self.leading_car.destroy()

            pygame.quit()

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
        location = self.leading_car.bounding_box.location
        location.y = location.y + self.leading_car.bounding_box.extent.y
        return location

    def get_leading_car_location(self):
        """
        Get the location of the back of the leading car
        :return: carla.Location(3D vector)
        """
        location = self.leading_car.bounding_box.location
        location.y = location.y - self.leading_car.bounding_box.extent.y
        return location


carla = CarlaConnection()
carla.main()
