import math

import carla
import easydict
import numpy as np
from FinalIntegration.Utils.navigation.custom_planner import AIonWheelsLocalPlanner
from FinalIntegration.Utils.navigation.global_route_planner import GlobalRoutePlanner


class BasicAgent():
    """Contains basic agent methods"""

    def __init__(self):
        pass

    def _vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass


class CarlaAgent(BasicAgent):
    def __init__(self, world, config=None) -> None:
        super().__init__()
        if config is None:
            config = dict()

        self._world = world
        self._vehicle = None
        self._map = self._world.get_map()
        self._waypoint_sampling_resolution = 2.0

        self._createHero()

        # planners:
        opt_dict = {}
        self._local_planner = AIonWheelsLocalPlanner(self._vehicle, opt_dict=opt_dict)
        self._global_planner = GlobalRoutePlanner(self._map, self._waypoint_sampling_resolution)

        # RL specific settings:
        self.actions = np.linspace(-1, 1, config.get('num_actions', 101))
        self._target_speed = config.get('target_speed', 15)  # m/s

        # self behavior
        self.behavior = easydict.EasyDict()
        self.behavior.ignore_speed_limit = False
        self.behavior.training = True

    def eval(self, evaluation=True, ignore_limit=False):
        self.behavior.training = not evaluation
        self.behavior.ignore_speed_limit = ignore_limit

    def train(self, training=True, ignore_limit=False):
        self.behavior.training = training
        self.behavior.ignore_speed_limit = ignore_limit

    def step(self, action: int, debug=False):
        assert (len(self.actions) > action >= 0, "Specified action out of bounds")

        if self.requires_plan():
            self.set_destination(np.random.choice(self._world.get_map().get_spawn_points()).location)
            print("Getting close to target, searching for next target")

        control = self._local_planner.run_step(debug)  # lateral control
        action = self.actions[action]
        if action > 0:
            control.throttle = action
        else:
            control.brake = -action  # need value between 0 and 1
        control.manual_gear_shift = False
        self._vehicle.apply_control(control)
        return control

    def _createHero(self):
        library = self._world.get_blueprint_library()
        blueprint = library.find('vehicle.tesla.model3')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = '217,96,65'
            blueprint.set_attribute('color', color)

        # spawn player vehicle
        spawn_points = self._world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
        self._vehicle = self._world.try_spawn_actor(blueprint, spawn_point)
        self._vehicle_physics(self._vehicle)

    # <editor-fold desc="Getters">
    def getVehicle(self):
        return self._vehicle

    def getTargetSpeed(self):
        if self.behavior.ignore_speed_limit:
            return self._target_speed
        elif self.behavior.training:
            return min(self._target_speed,
                       self._vehicle.get_speed_limit() * 1.6 / 3.6)  # increase speed limit for training, but avoid rediculus speeds causing crashes (however, 60 speed limits allow for >90!=> hopefully he learns the full range safe
        else:
            return min(self._target_speed, self._vehicle.get_speed_limit() / 3.6)

    def getSpeedLimit(self):
        return self._vehicle.get_speed_limit() / 3.6  # m/s

    def getSpeed(self):
        """Return the current velocity of the car in m/S"""
        vel = self._vehicle.get_velocity()
        return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def getVelocity(self):
        return self.getSpeed()

    def getWaypoints(self) -> list:
        return list(self._local_planner.get_plan())

    def getWidth(self):
        """returns half of the width"""
        return self._vehicle.bounding_box.extent.y

    def getHeight(self):
        """returns half of the height"""
        return self._vehicle.bounding_box.extent.z

    def getLength(self):
        """returns half of the length"""
        return self._vehicle.bounding_box.extent.x

    def getTransform(self):
        return self._vehicle.get_transform()

    # </editor-fold>

    # <editor-fold desc="Setters">
    def setTargetSpeed(self, speed):
        assert (speed >= 0, "Cannot reverse, negative speed is invalid")
        self._target_speed = speed

    # </editor-fold>

    def destroy(self):
        if self._vehicle:
            self._vehicle.destroy()
        self._vehicle = None

    # <editor-fold desc="pathfinding">
    def requires_plan(self):
        return len(self._local_planner.get_plan()) < 200

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            if len(self._local_planner.get_plan()) > 1:
                start_waypoint = self._local_planner.get_plan()[-1][0]
                clean_queue = False
            else:
                start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
                clean_queue = True
        elif len(self._local_planner.get_plan()) > 1:
            start_waypoint = self._local_planner.get_plan()[-1][0]
            clean_queue = False
        else:
            start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
            clean_queue = False

        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self._trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, stop_waypoint_creation=False, clean_queue=clean_queue)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)
    # </editor-fold>
