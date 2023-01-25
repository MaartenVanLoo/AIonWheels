import logging
import threading
import time

import carla

from FinalIntegration.Utils.CarlaAgent import CarlaAgent
from FinalIntegration.Utils.Sensor import Sensor, FollowCamera, Lidar, Camera, CollisionSensor,AsyncCamera,AsyncLidar, Radar
from FinalIntegration.Utils.TrafficGenerator import generateTraffic
from FinalIntegration.ReinforcementLearning.RL_Module import RL_Module
from .dist import distanceAlongPath
from ..DeepLearningLidar.DeepLearningLidar import DeeplearningLidar
from ..DeepLearningLidar.DeepLearningLidar_Distributed import DistributedLidar
from ..DeepLearningRecognition.DL_Recognition_Distributed import DistributedRecognition
from ..DeepLearningRecognition.DL_Recognition_module import DeepLearningRecognition

import numpy.random as random


def set_cuda_sync_mode(mode):
    """
    Set the CUDA device synchronization mode: auto, spin, yield or block.
    auto: Chooses spin or yield depending on the number of available CPU cores.
    spin: Runs one CPU core per GPU at 100% to poll for completed operations.
    yield: Gives control to other threads between polling, if any are waiting.
    block: Lets the thread sleep until the GPU driver signals completion.
    """
    import ctypes
    try:
        ctypes.CDLL('libcudart.dll').cudaSetDeviceFlags(
            {'auto': 0, 'spin': 1, 'yield': 2, 'block': 4}[mode])
    except Exception as e:
        print(e)
        print('Could not set cuda device flag')
        pass

class CarlaWorld(object):
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, 2000)
        self.client.set_timeout(100.0)
        print(self.client.get_available_maps())
        #self.client.load_world("Town10HD_Opt")
        self.client.load_world("Town01_Opt") #A basic town layout consisting of "T junctions".
        #self.client.load_world("Town02_Opt") #Similar to Town01, but smaller.
        #self.client.load_world("Town03_Opt") #The most complex town, with a 5-lane junction, a roundabout, unevenness, a tunnel, and more.
        #self.client.load_world("Town04_Opt") #An infinite loop with a highway and a small town.
        #self.client.load_world("Town05_Opt") #Squared-grid town with cross junctions and a bridge. It has multiple lanes per direction. Useful to perform lane changes.
        #self.client.load_world("Town06_Opt") #Long highways with many highway entrances and exits. It also has a Michigan left.
        #self.client.load_world("Town07_Opt") #	A rural environment with narrow roads, barns and hardly any traffic lights.

        self.debug = args.debug if 'debug' in args else False

        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager()
        self.fps = args.fps
        self._synchronous()

        self._player = None
        self.sensors = {}
        self.vehicle_list = []
        self.walker_list = []

        self.lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=1.80), carla.Rotation(pitch=0, yaw=0, roll=0))
        self.camera_transform = carla.Transform(carla.Location(x=1.4, y=0, z=1.60), carla.Rotation(pitch=0, yaw=0,roll=0))
        self.radar_transform = carla.Transform(carla.Location(x=2.4, y = 0, z=0.4), carla.Rotation(pitch=0, yaw=0, roll = 0))

        self.forced_start=args.forced_start
        self.emergency_brake=args.emergency_brake

        self._player = CarlaAgent(self.world, args)
        self._player.eval(ignore_limit=False)

        self.sensors["FollowCamera"] = FollowCamera(self._player.getVehicle(), self.world)
        self.sensors["CollisionSensor"] = CollisionSensor(self._player.getVehicle(), self.world)
        self.sensors["Radar"] = Radar(self._player.getVehicle(),self.world,self.radar_transform, self.args.radar_debug)
        if (args.MT):
            self.sensors["Lidar"] = AsyncLidar(self._player.getVehicle(), self.world, self.lidar_transform)
            self.sensors["Camera"] = AsyncCamera(self._player.getVehicle(), self.world, self.camera_transform)
        else:
            self.sensors["Lidar"] = Lidar(self._player.getVehicle(), self.world, self.lidar_transform)
            self.sensors["Camera"] = Camera(self._player.getVehicle(), self.world, self.camera_transform)

        self.emergency_brake_state = False

        # update world:
        self.world.tick()

        # AI modules:
        self.rl_module = RL_Module(self, args.rl_config)
        if (args.MT):
            self.dl_lidar = DistributedLidar(self, args.dl_lidar_config)
            self.dl_recognition = DistributedRecognition(self, args.dl_recognition_config)
        else:
            self.dl_lidar = DeeplearningLidar(self, args.dl_lidar_config)
            self.dl_recognition = DeepLearningRecognition(self, args.dl_recognition_config)

        #init recognition speed limit
        self.dl_recognition.current_max_speed = self.getPlayer().getSpeedLimit()

        # hud
        self.HUD = None



        #demo related variables
        self.crash_test = args.crash_test

    def getPlayer(self) -> CarlaAgent:
        return self._player


    def emergencyBrake(self,distance, action):
        if self.sensors["Radar"]:
            self.emergency_brake_state = self.sensors["Radar"].state
            return 0 if self.emergency_brake_state else action
        else:
            if distance<self.emergency_brake + 1/self.fps * self.getPlayer().getSpeed():
                self.emergency_brake_state = True
                return 0
            else:
                self.emergency_brake_state = False
                return action

    def step(self):
        start = time.time()
        for sensor in self.sensors.values():
            sensor.step()
        stop = time.time()
        print(f"Sensor update time:\t\t\t\t{(stop - start) * 1000:4.0f} ms")
        #print(f"Sensor update time:\t\t\t{(stop - start) * 1000:3.0f} ms")

        #update vision and lidar models
        self.dl_recognition.detect()
        self.dl_lidar.detect()

        #find distance based on lidar
        if self.dl_lidar.detected_boxes is None:
            distance = 110
        else:
            distance, _ = distanceAlongPath(
                self.getGlobalPath(),
                self.dl_lidar.detected_boxes,
                #self.getGlobalBoundingBoxes(),
                self.getPlayer().getWidth(),
                self.world,
                debug=self.debug
            )
        distance -= self._player.getLength() #correct distance for vehicle length
        distance += 10 if self.crash_test else 0

        #get parameters for RL model:
        red = self.dl_recognition.is_red_light
        orange = self.dl_recognition.is_orange_light
        green = not (red or orange)
        speed_limit = self.dl_recognition.current_max_speed if not self.dl_recognition.current_max_speed is None else \
            self.getPlayer().getSpeedLimit()


        action = self.rl_module.getAction(distance,speed_limit, red, orange, green)
        action = self.emergencyBrake(distance, action)
        self._player.step(action, debug=self.debug)

        start = time.time()
        self.world.tick()
        stop = time.time()
        print(f"Carla engine tick time:\t\t\t{(stop - start) * 1000:4.0f} ms")
        # update hud if required
        if self.HUD:
            self.HUD.render(self)

    def reset(self, map=None, layers=carla.MapLayer.All):
        # get variables to reset after reload
        vehicle_count = len(self.vehicle_list)
        walker_count = len(self.walker_list)

        hud_backup = self.HUD

        # get loaded sensors:
        sensor_ids = []
        for sensor in self.sensors.keys():
            sensor_ids.append(sensor)

        # destroy current actors
        self.destroy()

        # reset carla
        if map is None or map not in self.client.get_available_maps():
            maps = self.client.get_available_maps()
            maps = [map for map in maps if "Opt" in map] #filter for opt maps
            map = random.choice(maps)
        maps = ["Town03_Opt", "Town04_Opt"]
        print(f"Loading map: {map}")
        self.client.load_world(map, reset_settings=False, map_layers=layers)
        self.world = self.client.get_world()


        self._synchronous()

        # rebuild sensors & agent:
        self._player = CarlaAgent(self.world, self.args)
        self._player.eval()
        self.world.tick() #make sure the agent is created?
        for sensor in sensor_ids:
            if sensor == "FollowCamera":
                self.sensors["FollowCamera"] = FollowCamera(self._player.getVehicle(), self.world)
            elif sensor == "CollisionSensor":
                self.sensors["CollisionSensor"] = CollisionSensor(self._player.getVehicle(), self.world)
            elif sensor == "Lidar":
                self.sensors["Lidar"] = AsyncLidar(self._player.getVehicle(), self.world, self.lidar_transform) if (self.args.MT) \
                                   else Lidar(self._player.getVehicle(), self.world, self.lidar_transform)
            elif sensor == "Camera":
                self.sensors["Camera"] = AsyncCamera(self._player.getVehicle(), self.world, self.camera_transform) if (self.args.MT) \
                                    else Camera(self._player.getVehicle(), self.world, self.camera_transform)
            elif sensor == "Radar":
                self.sensors["Radar"] = Radar(self._player.getVehicle(),self.world,self.radar_transform,self.args.radar_debug)
            else:
                print(f'Cannot recreate sensor {sensor}')

        # respawn traffic
        self.spawn(vehicle_count, walker_count)

        # restore hud
        if hud_backup:
            self.attachHUD(hud_backup)

        #restore module agent references:
        self.rl_module._agent = self._player
        self.dl_lidar._agent = self._player
        self.dl_recognition._agent = self._player

        for i in range(self.forced_start):
            self.rl_module.getAction(100)
            self._player.step(len(self._player.actions)-1, debug=self.debug)
            self.world.tick()

    def _synchronous(self):
        # Set Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.fps
        settings.no_rendering_mode = False  # otherwise gpu sensors will stop working
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(True)
        self.world.tick()
        self.map = self.world.get_map()

    def _asynchronous(self):
        # Set asynchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
        settings.no_rendering_mode = False  # otherwise gpu sensors will stop working
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(False)
        self.world.wait_for_tick()

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
        # unbind hud
        self.disableHUD()

        self.dl_recognition.cleanup()
        self.dl_lidar.cleanup()


    def spawn(self, vehicles=50, walkers=0):
        if walkers > 0:
            UserWarning("Walkers are not yet supported")
        self.vehicle_list, self.walker_list = generateTraffic(self.world, self.client, self.traffic_manager,
                                                              number_of_vehicles=vehicles, number_of_walkers=walkers)

    def get_sensor(self, name="") -> Sensor:
        return self.sensors.get(name, None)

    def attachHUD(self, hud):
        if not hud.id == -1:
            print("Could not attach HUD. HUD is already attached to another world")
            return
        self.HUD = hud
        self.HUD.carlaWorld = self
        self.HUD.id = self.world.on_tick(self.HUD.on_world_tick)

    def disableHUD(self):
        if self.HUD:
            self.world.remove_on_tick(self.HUD.id)
            self.HUD.id = -1
            self.HUD.carlaWorld = None
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
