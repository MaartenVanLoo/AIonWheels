"""
Custom local carla planner
This planner only has automatic lateral control,
Longitudinal control must be handled externaly
"""
import time
from collections import deque

import numpy.random as random

import carla

from .local_planner import RoadOption
from .controller import PIDLateralController
from .tools.misc import draw_waypoints, get_speed


class AIonWheelsLocalPlanner(object):
    def __init__(self, vehicle, opt_dict={}):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with different parameters:
            dt: time between simulation steps
            target_speed: desired cruise speed in Km/h
            sampling_radius: distance between the waypoints part of the plan
            lateral_control_dict: values of the lateral PID controller
            max_throttle: maximum throttle applied to the vehicle
            max_brake: maximum brake applied to the vehicle
            max_steering: maximum steering applied to the vehicle
            offset: distance between the route waypoints and the center of the lane
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._vehicle_controller = None
        self.target_waypoint = None
        self.target_road_option = None

        self._waypoints_queue = deque(maxlen=10000)
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False

        # Base parameters
        self._dt = 1.0 / 20.0
        # self._target_speed = 20.0  # Km/h  #not used
        self._sampling_radius = 3.0
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self._dt}
        # self._max_throt = 0.75
        # self._max_brake = 0.3
        self._max_steer = 0.8
        self.past_steering = self._vehicle.get_control().steer
        self._offset = 0
        self._base_min_distance = 3.0
        self._follow_speed_limits = False

        # Overload default parameters
        if opt_dict:
            if 'lateral_control_dict' in opt_dict:
                self._args_lateral_dict = opt_dict['lateral_control_dict']

        # initializing controller
        self._init_controller()


        #line drawing variabels
        self._current_frame = 0

    def _init_controller(self):
        """Controller initialization"""
        self._vehicle_controller = PIDLateralController(self._vehicle,
                                                        offset=self._offset,
                                                        **self._args_lateral_dict)

        # Compute the current vehicle waypoint
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def reset_vehicle(self):
        """Reset the ego-vehicle"""
        self._vehicle = None

    def run_step(self, debug=False):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """
        # Add more waypoints too few in the horizon
        if not self._stop_waypoint_creation and len(self._waypoints_queue) < self._min_waypoint_queue_length:
            self._compute_next_waypoints(k=self._min_waypoint_queue_length)

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vehicle_speed = get_speed(self._vehicle) / 3.6
        self._min_distance = self._base_min_distance + 0.4 * vehicle_speed

        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:

            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # Get the target waypoint and move using the PID controllers. Stop if no target waypoint
        control = carla.VehicleControl()
        if len(self._waypoints_queue) == 0:

            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = True
            control.manual_gear_shift = False
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            current_steering = self._vehicle_controller.run_step(self.target_waypoint)

            if current_steering > self.past_steering + 0.1:
                current_steering = self.past_steering + 0.1
            elif current_steering < self.past_steering - 0.1:
                current_steering = self.past_steering - 0.1
            if current_steering >= 0:
                steering = min(self._max_steer, current_steering)
            else:
                steering = max(-self._max_steer, current_steering)
            control.steer = steering
            control.hand_brake = False
            control.manual_gear_shift = False
            self.past_steering = steering

        #if debug:
        #    draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], 1.0)

        self._current_frame -= 1
        if self._current_frame <= 0 and debug: #once a second!
            self._current_frame = 20
            _draw_path(self._vehicle.get_world(), self._waypoints_queue)
        #print(f"Current queue length: {len(self._waypoints_queue)}")
        return control

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def get_plan(self):
        """Returns the current plan of the local planner"""
        return self._waypoints_queue

    def done(self):
        """
        Returns whether or not the planner has finished

        :return: boolean
        """
        return len(self._waypoints_queue) == 0

    def set_global_plan(self, current_plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs
        The 'clean_queue` parameter erases the previous plan if True, otherwise, it adds it to the old one
        The 'stop_waypoint_creation' flag stops the automatic creation of random waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :param stop_waypoint_creation: bool
        :param clean_queue: bool
        :return:
        """
        if clean_queue:
            self._waypoints_queue.clear()

        # Remake the waypoints queue if the new plan has a higher length than the queue
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue

        if len(self._waypoints_queue) == 0:
            for elem in current_plan:
                self._waypoints_queue.append(elem)
        else:
            for elem in current_plan:
                if not self._waypoints_queue[-1] == elem:
                    #compute angle between waypoints:
                    w = self._waypoints_queue[-1][0].transform.location     #waypoint
                    w_1 = self._waypoints_queue[-2][0].transform.location   #prev waypoint
                    w1 = elem[0].transform.location                         #next waypoint
                    v1 = w - w_1
                    v2 = w1 - w
                    while v1.dot(v2) < 0:
                        self._waypoints_queue.pop()
                        w = self._waypoints_queue[-1][0].transform.location
                        w_1 = self._waypoints_queue[-2][0].transform.location  # next waypoint
                        v1 = w - w_1
                        v2 = w1 - w
                        print(f"Removed sharp corner from route while setting new target")
                    self._waypoints_queue.append(elem)
                else:
                    self._waypoints_queue.pop() # current plan wants to take this node => hence removing it from both
                    # the plan and the waypoints is still a valid route
                    print(f"Removed node from waypoint queue while setting new target")




        self._stop_waypoint_creation = stop_waypoint_creation

def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT



def _draw_path(world, path, fading = .95):
    offset = carla.Location(0.1)
    points = [p[0].transform.location+offset for p in path]
    alpha = 20
    for i in range(len(points)-1):
        begin = points[i]
        end = points[i+1]
        begin.z += 0.1
        end.z += 0.1
        color = carla.Color(0,5,0,a=int(alpha))
        world.debug.draw_line(begin, end, thickness = 1, color = color, life_time = 1.1)

        # update alpha
        alpha *= fading
        if (alpha < 1):
            return  # limited lenthg
