

"""
Class implementing an interface for our Carla Agent
"""
import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import get_speed
from CustomPlanners import AIonWheelsLocalPlanner
import numpy as np
class CarlaAgent(object):
    def __init__(self, vehicle) -> None:
        super().__init__()


class CarlaAgentRL(object):
    def __init__(self, vehicle, num_actions: int, target_speed=20, opt_dict={}) -> None:
        super().__init__()
        self.actions = np.linspace(-1, 1, num_actions)
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False

        #self._target_speed = target_speed
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._max_brake = 0.5

        # Change parameters according to the dictionary
        opt_dict['target_speed'] = target_speed
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'max_brake' in opt_dict:
            self._max_steering = opt_dict['max_brake']

        # Initialize the planners
        self._local_planner = AIonWheelsLocalPlanner(self._vehicle, opt_dict=opt_dict)
        self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

    def run_step(self, action: int):
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        vehicle_speed = get_speed(self._vehicle) / 3.6  # m/s

        control = self._local_planner.run_step() # lateral control

        action = self.actions[action]
        if action > 0:
            control.throttle = action
        else:
            control.brake = -action #need value between 0 and 1

        return control

    def getPos(self):
        return self._world.get_actor(self._vehicle.id).get_location()

    def getVel(self):
        return self._world.get_actor(self._vehicle.id).get_velocity()

    def getTransform(self):
        return self._world.get_actor(self._vehicle.id).get_transform()
    def getBBox(self):
        return self._world.get_actor(self._vehicle.id).bounding_box

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
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def destroy(self):
        carla.command.DestroyActor(self._vehicle)