"""
This file contains the main world class used as top level API to connect with carla
"""
import argparse
import math
import time
import traceback

import numpy.random as random
import numpy as np
from CarlaWorld import HUD, World, KeyboardControl
import CarlaAgents

import carla
import pygame
import weakref

from ReinforcementLearning.CarlaEnvironment import TrafficGenerator
from ReinforcementLearning.CarlaEnvironment.utils import dist
from ReinforcementLearning.CarlaEnvironment.utils.ClientSideBouningBoxes import ClientSideBoundingBoxes


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

        #sensors:
        self.collision_sensor = None
        self.lidar_sensor = None

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

    def spawnVehicles(self, number_of_vehicles=50, args=None):
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
        ######agent_transform = self.agent.getTransform()
        ######agent_dir = agent_transform.get_forward_vector()
        ######agent_dir.z = 0
        ######agent_dir = agent_dir.make_unit_vector()
        wheel_vector = -self.agent._vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
        wheel_vector = np.radians(wheel_vector)
        agent_transform = self.agent._vehicle.get_transform()
        agent_dir = agent_transform.get_forward_vector()
        agent_dir.z = 0
        agent_dir = agent_dir.make_unit_vector()
        aux = carla.Vector3D()
        aux.x = agent_dir.x
        aux.y = agent_dir.y
        aux.z = agent_dir.z
        agent_dir.x = (aux.x * np.cos(wheel_vector)) - (aux.y * np.sin(wheel_vector))
        agent_dir.y = (aux.x * np.sin(wheel_vector)) + (aux.y * np.cos(wheel_vector))

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
        sim = self.client.get_world()
        self.settings = sim.get_settings()
        self.settings.synchronous_mode = False
        self.traffic_manager.set_synchronous_mode(False)
        sim.apply_settings(self.settings)

        maps = self.client.get_available_maps()
        #self.client.load_world(random.choice(maps), reset_settings=False)
        self.client.load_world("Town04_Opt", reset_settings=False)
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
            self._drawBoundingBoxes()
            pygame.display.flip()

        #update agent
        if self.agent.done():
            self.agent.set_destination(random.choice(self.spawn_points).location)
            self.world.hud.notification("The target has been reached, searching for another target", seconds=4.0)
            print("The target has been reached, searching for another target")
        if self.agent.requires_plan():
            self.agent.set_destination(random.choice(self.spawn_points).location)
            self.world.hud.notification("Getting close to target, searching for next target",
                                        seconds=4.0)
            print("Getting close to target, searching for next target")

        control = self.agent.run_step(action=action)
        control.manual_gear_shift = False
        self.world.player.apply_control(control)
        return control

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def addCollisionSensor(self):
        if self.collision_sensor:
            return
        self.collision_sensor = self._CustomCollisionSensor(self.world.player, self.on_collision)

    def getCollisionSensor(self):
        return self.collision_sensor

    def getCollisionIntensity(self):
        if not self.collision_sensor:
            return None
        return self.collision_sensor.getIntensity()

    def addLidarSensor(self):
        if self.lidar_sensor:
            return
        self.lidar_sensor = self._CustomLidarSensor(self.world.player)

    def getLidarSensor(self):
        return self.lidar_sensor


    def getDistanceAlongPath(self, debug = False):
        """
        Distance measured allong the waypoint path of the actor
        """
        actors = self.world.world.get_actors(self.vehicles_list)

        bb = [actor.bounding_box for actor in actors]
        transforms = [actor.get_transform() for actor in actors]

        agent = self.agent._vehicle
        agent_tt = agent.get_transform()
        agent_bb = agent.bounding_box

        invRotation = carla.Transform(
            carla.Location(0,0,0),
            carla.Rotation(-agent_tt.rotation.pitch, -agent_tt.rotation.yaw, -agent_tt.rotation.roll))

        waypoints = self.agent.getWaypoints()
        waypoints = [waypoint[0] for waypoint in waypoints] #only filter out the location

        #agent space = center of agent bounding box
        # change all boxes to "agent space"
        for box, transform in zip(bb, transforms):
            box.location += transform.location - agent_tt.location - agent_bb.location
            # correct the global angle with the agent angle
            box.rotation.yaw =   transform.rotation.yaw   #- agent_tt.rotation.yaw
            box.rotation.pitch = transform.rotation.pitch #- agent_tt.rotation.pitch
            box.rotation.roll =  transform.rotation.roll  #- agent_tt.rotation.roll

            #TODO:scale boxes with the size of the agent box to incorporate the non zero size of the agent
            #TODO:check orientation
            #box.extent += agent_bb.extent

        # change all waypoints to "agent_space":

        agent_waypoint =  carla.Location(agent_tt.location.x, agent_tt.location.y, agent_tt.location.z)
        agent_waypoint.z = waypoints[0].transform.location.z
        transformed_waypoints = []#[agent_waypoint - agent_tt.location + agent_bb.location] #first point = agent
                                 # bounding box
        for waypoint in waypoints:
            vec = waypoint.transform.location
            vec.x += -agent_tt.location.x + agent_bb.location.x
            vec.y += -agent_tt.location.y + agent_bb.location.y
            #invRotation = carla.Transform(carla.Location(0,0,0), carla.Rotation(
            #      -agent_tt.rotation.pitch,             #-waypoint.transform.rotation.pitch
            #        -agent_tt.rotation.yaw,             #-waypoint.transform.rotation.yaw
            #       -agent_tt.rotation.roll))                #-waypoint.transform.rotation.roll
            #vec = invRotation.transform(vec)
            transformed_waypoints.append(carla.Location(vec.x, vec.y, vec.z))
        agent_waypoint = carla.Location(0,0,transformed_waypoints[0].z)
        transformed_waypoints.insert(0,agent_waypoint)

        z_offset = transformed_waypoints[0].z # first waypoint must be zero, others adapt accordingly
        for waypoint in transformed_waypoints:
            waypoint.z -= z_offset

        distance = dist.distanceAlongPath(transformed_waypoints, bb, agent_bb.extent.y, self.world, debug=debug)
        #correct distance for own car length
        distance -= agent_bb.extent.x
        return distance

    def debug(self):
        """
        some debug code
        """
        #get all actors:
        actors = self.world.world.get_actors(self.vehicles_list)
        bb = [actor.bounding_box for actor in actors]
        transforms = [actor.get_transform() for actor in actors]

        agent = self.agent._vehicle
        agent_bb = agent.bounding_box
        agent_tt = agent.get_transform()


    def _drawBoundingBoxes(self):
        actors = self.world.world.get_actors().filter(
            'vehicle.*')
        agent = self.agent._vehicle
        transforms = [actor.get_transform() for actor in actors]
        agent_tt = agent.get_transform()

        close_actors = []
        for idx,actor in enumerate(actors):
            if (agent_tt.location.distance(transforms[idx].location) < 50):
                close_actors.append(actor)

        # draw bounding boxes close to agent:
        cam_id = self.world.camera_manager.sensor.id
        bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self,close_actors, self.world.world.get_actor(cam_id))
        ClientSideBoundingBoxes.draw_bounding_boxes(self,self.display, bounding_boxes)


    def setTrafficLights(self,state:carla.TrafficLightState, duration :float= None) -> None:
        if not duration or duration < 0:
            duration = 99999.0
        list_actor = self.world.world.get_actors()
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                # for any light, first set the light state, then set time. for yellow it is
                # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
                actor_.set_state(state)
                actor_.set_green_time(duration)




    @staticmethod
    def on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.intensity = intensity

    class _CustomCollisionSensor(object):
        def __init__(self, parent_actor, on_collision):
            self.sensor = None
            self._parent = parent_actor
            self.intensity = 0
            world = self._parent.get_world()
            blueprint = world.get_blueprint_library().find('sensor.other.collision')
            self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: on_collision(weak_self, event))

        def getIntensity(self):
            return self.intensity

    class _CustomCameraSensor(object):
        pass

    class _CustomLidarSensor(object):
        def __init__(self,parent_actor):
            self.sensor = None
            self._parent = parent_actor
            self.intensity = 0
            world = self._parent.get_world()
            lidar_bp  = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('rotation_frequency',"20") #equal to 1/time per step
            lidar_bp.set_attribute('range', "80")
            lidar_bp.set_attribute('points_per_second', "2500000")
            #lidar_bp.set_attribute('sensor_tick', "0.05")
            lidar_bp.set_attribute('channels', "64")
            lidar_bp.set_attribute('upper_fov', "2")
            lidar_bp.set_attribute('lower_fov', "-24.8")
            lidar_bp.set_attribute('dropoff_general_rate',
                                   lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit',
                                   lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity',
                                   lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
            lidar_location = carla.Location(x=0,z=2.8)
            lidar_transform = carla.Transform(lidar_location)
            self.sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=self._parent)
            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: self._on_frame(weak_self, event))
            pass

        def _on_frame(self,weak_self, event):
            event.save_to_disk("test.ply")
            np.frombuffer(event.raw_data,dtype=np.float32).tofile("test.bin")
            print(np.frombuffer(event.raw_data,dtype=np.float32))
            print(f'Lidar angle:{event.horizontal_angle}')
            print(f'Lidar frame:{event.frame}')
            print(f'Lidar time:{event.timestamp}')
            print(f'Lidar:{event.get_point_count(0)}')
            pass

        def getRawData(self):
            if not self.sensor.raw_data:
                return None
            return self.sensor.raw_data





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
        worldapi.reset() #load random map
        #worldapi.spawnVehicles(args={'vehicle_filter': 'vehicle.tesla.cybertruck'})
        worldapi.spawnVehicles(50)
        worldapi.addAgent(CarlaAgents.CarlaAgentRL(worldapi.world.player, num_actions=11))
        worldapi.addCollisionSensor()
        #worldapi.addLidarSensor()

        #worldapi.setTrafficLights(carla.TrafficLightState.Green)

        print(worldapi.agent.getPos())
        frame = 0
        while True:
            frame += 1
            if frame%3 == 0:
                DIST = worldapi.getDistanceAlongPath(debug=True) #draw lines, not every frame to improve performance
            else:
                DIST = worldapi.getDistanceAlongPath()
            print(f"Distance:{DIST}")
            action = int(DIST-worldapi.agent.getVel().length() - 7)

            action = min(max(int(action),0),8) #clip action
            worldapi.step(action=action)
            #worldapi.getClosestVechicle()
            #print(worldapi.getCollisionIntensity())
            #time.sleep(0.1)

    except:
        traceback.print_exc()
    finally:
        if worldapi:
            worldapi.cleanup()

    pass
