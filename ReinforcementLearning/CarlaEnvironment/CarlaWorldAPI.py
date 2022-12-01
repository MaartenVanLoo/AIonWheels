"""
This file contains the main world class used as top level API to connect with carla
"""
import argparse
import traceback

import numpy.random as random
import numpy as np
from CarlaWorld import HUD, World, KeyboardControl
import CarlaAgents

import carla
import pygame

from ReinforcementLearning.CarlaEnvironment import TrafficGenerator


class CarlaWorldAPI:
    def __init__(self, args, host='127.0.0.1', port=2000, width=1280, height=720, fullscreen=False, show=True) \
            -> None:
        super().__init__()
        pygame.init()
        pygame.font.init()
        self.world = None
        self.host = host
        self.port = port

        self.args = args
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        self.traffic_manager = self.client.get_trafficmanager()
        self.sim_world = self.client.get_world()

        # use synchronus mode:
        self.settings = self.sim_world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        self.sim_world.apply_settings(self.settings)
        self.traffic_manager.set_synchronous_mode(True)

        self.clock = pygame.time.Clock()
        # open display if needed:
        self.full_screen = fullscreen
        self.width = width
        self.height = height
        if show:
            self.__show(width, height)
        else:
            self.__hide()

        self.hud = HUD(width, height)
        self.world = World(self.client.get_world(), self.hud, args)
        self.controller = KeyboardControl(self.world)

        self.spawn_points = self.world.map.get_spawn_points()

        self.agent = None

        self.vehicles_list = []
        self.number_of_vehicles = -1

    def cleanup(self):
        print("Cleaning up world")
        if self.world is not None:
            print("Resetting world settings")
            settings = self.world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

            if self.vehicles_list:
                print('Destroying %d vehicles' % len(self.vehicles_list))
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
            else:
                print("No vehicles found")

            print("destroying world object")
            self.world.destroy()
        print("quit pygame")
        pygame.quit()
        print("Cleanup done")

    def __show(self, width=1280, height=720):
        options = pygame.HWSURFACE | pygame.DOUBLEBUF
        if self.full_screen:
            options |= pygame.FULLSCREEN

        self.display = pygame.display.set_mode(
            (width, height),
            options)

    def __hide(self):
        self.display = None

    def addAgent(self, agent: object):
        self.agent = agent
        origin = random.choice(self.spawn_points).location
        # Set the agent destination

        destination = random.choice(self.spawn_points).location
        agent.set_destination(end_location=destination)

        pass

    def spawnVehicles(self, number_of_vehicles=30, args=None):
        if args is None:
            args = dict()
        if self.vehicles_list:
            print('Destroying %d vehicles' % len(self.vehicles_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.vehicles_list = TrafficGenerator.generateTraffic(self.world.world, self.client, self.traffic_manager,
                                                              number_of_vehicles=number_of_vehicles,
                                                              args=args)
        self.number_of_vehicles = number_of_vehicles

    def getClosestVechicle(self):
        """
        Get the vehicle id and distance to the closest car in front of the agent
        lead_id = None if no vehicle within range
        distance = np.Inf if no vehicle within range
        @returns tuple containing (lead_id, distance)
        """""
        positions = []
        bbox = []
        for actor in self.world.world.get_actors(self.vehicles_list):
            positions.append(actor.get_transform())
            bbox.append(actor.bounding_box)
        agent_transform = self.agent.getTransform()
        agent_dir = agent_transform.get_forward_vector()
        agent_dir.z = 0
        agent_dir = agent_dir.make_unit_vector()

        angles = []
        index = -1
        best_dist = np.Inf
        for i, p in enumerate(positions):
            dist = p.location.distance(agent_transform.location)
            diff = p.location - agent_transform.location
            diff.z = 0
            diff = diff.make_unit_vector()
            dot = np.clip(agent_dir.dot_2d(diff), -1, 1)
            angles.append(np.arccos(dot))
            if abs(angles[-1]) <= 0.12:  # ±6°
                if dist < best_dist:  # correct dist with angle, larger angle => bigger virtual distance to avoid
                    # distances from nearby lanes, directly in front, dot = 1!
                    best_dist = dist
                    index = i
        # correct for bounding box size
        if not index == -1:
            best_dist = best_dist - bbox[index].extent.x - self.agent.getBBox().extent.x
        vehicle_id = None if index == -1 else self.vehicles_list[index]
        return vehicle_id, abs(best_dist)

        '''
        for actor in self.world.world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'hero':
                print("PLAYER")

            print(actor.get_location())
        '''

    def getVehicleSpeed(self, vehicleId):
        return self.world.world.get_actor(vehicleId).get_velocity()

    def reset(self):
        # cleanup agent & AI vehicles
        if self.agent:
            self.agent.destroy()

        if self.vehicles_list:
            print('Destroying %d vehicles' % len(self.vehicles_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        else:
            print("No vehicles found")

        # no synchronous mode, reloading world not possible when synchronous
        sim = self.client.get_world();
        self.settings = sim.get_settings()
        self.settings.synchronous_mode = False
        self.traffic_manager.set_synchronous_mode(False)
        sim.apply_settings(self.settings)

        maps = self.client.get_available_maps()
        self.client.load_world(random.choice(maps), reset_settings=False)
        # self.client.reload_world(reset_settings=False)

        # reset synchronous mode and reload GUI elements
        self.sim_world = self.client.get_world()
        self.settings = self.sim_world.get_settings()
        self.settings.synchronous_mode = True
        self.traffic_manager = self.client.get_trafficmanager()
        self.settings.fixed_delta_seconds = 0.05
        self.sim_world.apply_settings(self.settings)

        self.hud = HUD(self.width, self.height)
        self.world = World(self.client.get_world(), self.hud, self.args)
        self.controller = KeyboardControl(self.world)

        self.agent = None
        self.spawn_points = self.world.map.get_spawn_points()

        # respawn vehicles
        if self.number_of_vehicles > 0:
            self.spawnVehicles(self.number_of_vehicles)
        print(self.world.world.get_settings())

        pass

    def step(self, action: int):
        # make the world move
        self.clock.tick()
        self.world.world.tick()
        self.world.tick(self.clock)
        if self.controller.parse_events():
            return
        if self.display:
            self.world.render(self.display)
            pygame.display.flip()

        # update agent
        if self.agent.done():
            self.agent.set_destination(random.choice(self.spawn_points).location)
            self.world.hud.notification("The target has been reached, searching for another target", seconds=4.0)
            print("The target has been reached, searching for another target")

        control = self.agent.run_step(action=action)
        control.manual_gear_shift = False
        self.world.player.apply_control(control)
        return control

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


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
    worldapi = None
    try:
        worldapi = CarlaWorldAPI(args=args)
        worldapi.spawnVehicles(args={'vehicle_filter': 'vehicle.tesla.cybertruck'})
        worldapi.addAgent(CarlaAgents.CarlaAgentRL(worldapi.world.player, num_actions=11))
        print(worldapi.agent.getPos())
        while True:
            worldapi.step(action=10)

    except:
        traceback.print_exc()
    finally:
        if worldapi:
            worldapi.cleanup()

    pass
