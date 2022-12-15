import logging
import threading
import time

import carla

from FinalIntegration.Utils.CarlaAgent import CarlaAgent
from FinalIntegration.Utils.Sensor import Sensor,FollowCamera, Lidar, Camera, CollisionSensor
from FinalIntegration.Utils.TrafficGenerator import generateTraffic
from FinalIntegration.ReinforcementLearning.RL_Module import  RL_Module
from .dist import distanceAlongPath
from ..DeepLearningLidar.DeepLearningLidar import DeeplearningLidar
from ..DeepLearningRecognition.DeepLearningRecognition import DeepLearningRecognition


class CarlaWorld(object):
    def __init__(self, args):
        self.client = carla.Client(args.host, 2000)
        self.client.set_timeout(100.0)
        self.client.load_world("Town10HD_Opt")
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager()
        self.fps = args.fps
        self._synchronous()

        self._player = None
        self.sensors = {}
        self.vehicle_list = []
        self.walker_list = []

        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=1.80), carla.Rotation(pitch=0, yaw=0, roll=0))
        camera_transform = carla.Transform(carla.Location(x=1.1, y=0, z=1), carla.Rotation(pitch=0, yaw=0, roll=0))

        self._player = CarlaAgent(self.world, args)
        self.sensors["FollowCamera"] = FollowCamera(self._player.getVehicle(), self.world)
        self.sensors["CollisionSensor"] = CollisionSensor(self._player.getVehicle(), self.world)
        self.sensors["Lidar"] = Lidar(self._player.getVehicle(), self.world, lidar_transform)
        self.sensors["Camera"] = Camera(self._player.getVehicle(), self.world, camera_transform)

        # update world:
        self.world.tick()

        # AI modules:
        self.rl_module = RL_Module(self, args.rl_config)
        self.dl_lidar = DeeplearningLidar(self, args.dl_lidar_config)
        self.dl_recognition = DeepLearningRecognition(self, args.dl_recognition_config)


        # hud
        self.HUD = None

        self.debug = args.debug  if 'debug' in args else False

    def getPlayer(self) ->CarlaAgent:
        return self._player


    def step(self):
        start = time.time()
        for sensor in self.sensors.values():
            sensor.step()
        stop = time.time()
        print(f"Sensor update time:\t\t\t{(stop - start) * 1000:3.0f} ms")
        distance = self.dl_lidar.getDistance()
        distance, _ = distanceAlongPath(
            self.getGlobalPath(),
            self.getGlobalBoundingBoxes(),
            self.getPlayer().getWidth(),
            self.world,
            debug = self.debug
        )
        distance -= self._player.getLength()
        action = self.rl_module.getAction(distance)
        self._player.step(action, debug = False)


        start = time.time()
        self.world.tick()
        stop = time.time()
        print(f"Carla engine tick time:\t\t{(stop - start) * 1000:3.0f} ms")
        #update hud if required
        if self.HUD:
            self.HUD.render(self)



    def _synchronous(self):
        # Set Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.fps
        settings.no_rendering_mode = False  # otherwise gpu sensors will stop working
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(True)

    def _asynchronous(self):
        # Set asynchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
        settings.no_rendering_mode = False  # otherwise gpu sensors will stop working
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(False)

    def destroy(self):
        if self._player:
            self._player.destroy()
        for sensor in self.sensors.values():
            sensor.destroy()
        self.sensors = {}

        # remove vehicles:
        if self.vehicle_list:
            print('Destroying %d vehicles' % len(self.vehicle_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])
        else:
            print("No vehicles found")
        self.vehicle_list = []
        # remove walkers
        if self.walker_list:
            print('Destroying %d walkers' % len(self.walker_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
        else:
            print("No walker found")
        self.walker_list = []
        self._asynchronous()

    def spawn(self, vehicles=50, walkers=0):
        if walkers > 0:
            UserWarning("Walkers are not yet supported")
        self.vehicle_list, self.walker_list = generateTraffic(self.world, self.client, self.traffic_manager,
                                                              number_of_vehicles=vehicles, number_of_walkers=walkers)

    def get_sensor(self, name="") ->Sensor:
        return self.sensors.get(name, None)

    def attachHUD(self,hud):
        if not hud.id == -1:
            print("Could not attach HUD. HUD is already attached to another world")
            return
        self.HUD = hud
        self.HUD.id = self.world.on_tick(self.HUD.on_world_tick)

    def disableHUD(self):
        if self.HUD:
            self.world.remove_on_tick(self.HUD.id)
            self.HUD.id = -1
            self.HUD = None

    def getGlobalPath(self):
        agent = self._player.getVehicle()
        agent_tt = agent.get_transform()
        agent_bb = agent.bounding_box

        waypoints = self._player.getWaypoints()
        waypoints = [waypoint[0] for waypoint in waypoints]  # only filter out the location

        """Convert to global space"""

        # waypoints => to center of bounding box
        transformed_waypoints = []
        for waypoint in waypoints:
            z = waypoint.transform.location.z + agent_bb.location.z
            transformed_waypoints.append(carla.Location(waypoint.transform.location.x, waypoint.transform.location.y,
                                                        z))
        agent_waypoint = agent_tt.location + agent_bb.location
        transformed_waypoints.insert(0, carla.Location(agent_waypoint.x, agent_waypoint.y, agent_waypoint.z))
        return transformed_waypoints
    def getGlobalBoundingBoxes(self):
        actors = self.world.get_actors(self.vehicle_list)

        bb = [actor.bounding_box for actor in actors]
        transforms = [actor.get_transform() for actor in actors]

        for box, transform in zip(bb, transforms):
            # from vehicle to global space"""
            box.location += transform.location
            box.rotation.pitch += transform.rotation.pitch
            box.rotation.yaw += transform.rotation.yaw
            box.rotation.roll += transform.rotation.roll
        return bb
