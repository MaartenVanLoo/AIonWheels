#build this module using cli:
# python setup.py build_ext --inplace

import math

import carla

from libcpp cimport bool
from libc.math cimport sqrt
import numpy as np

cdef class Vector3D:
    cdef public double x
    cdef public double y
    cdef public double z

    def __cinit__(self, double x_, double y_, double z_):
        self.x = x_
        self.y = y_
        self.z = z_

    def __sub__(self, other: Vector3D):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    def __add__(self, other : Vector3D):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    def __truediv__(self, other: Vector3D):
        return Vector3D(self.x / other.x, self.y / other.y, self.z / other.z)

    cdef double getMax(self):
        return max(max(self.x,self.y),self.z)
    cdef double getMin(self):
        return min(min(self.x,self.y),self.z)

    @staticmethod
    cdef Vector3D max_(v1:Vector3D, v2: Vector3D):
        return Vector3D(max(v1.x, v2.x),max(v1.y, v2.y), max(v1.z, v2.z))
    @staticmethod
    cdef Vector3D min_(v1: Vector3D, v2: Vector3D):
        return Vector3D(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z))


cdef class Location(Vector3D):
    def __cinit__(self, double x_, double y_,double z_):
        self.x = x_
        self.y = y_
        self.z = z_

    def __sub__(self, Location other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, Location other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    cdef double distance_sq(self, Location other):
        return (self.x - other.x) *  (self.x - other.x)+\
               (self.y - other.y) *  (self.y - other.y)+\
               (self.z - other.z) *  (self.z - other.z)

    cdef double distance(self, Location other):
        return sqrt(self.distance_sq(other))

cdef class Rotation:
    cdef public double pitch
    cdef public double roll
    cdef public double yaw

    def __cinit(self, float pitch, float roll, float yaw):
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw


cdef class BoundingBox(object):
    cdef public Vector3D location
    cdef public Vector3D extent
    cdef public Rotation rotation

    #todo fill in this class
    def __cinit__(self,Location pos, Vector3D extent, Rotation rotation_  ):
        self.location = Vector3D(pos.x, pos.y, pos.z)
        self.extent = Vector3D(extent.x, extent.y, extent.z)
        self.rotation = rotation_

cdef class Ray(object):
    cdef public Vector3D origin
    cdef public Vector3D direction

    def __cinit__(self, Vector3D start,Vector3D end) -> None:
        super().__init__()
        #print(f"Ray start:{start.x}, {start.y}, {start.z}")
        #print(f"Ray end  :{end.x}, {end.y}, {end.z}")
        self.origin = Vector3D(start.x,start.y,start.z)
        self.direction = Vector3D(end.x - start.x,end.y - start.y,end.z - start.z)

    def draw(self,world):
        #world.debug.draw_line(self.origin, self.origin + self.direction)
        t = world.player.get_transform()
        b = world.player.bounding_box
        o = carla.Location(self.origin.x,self.origin.y,self.origin.z)
        d = carla.Location(self.direction.x,self.direction.y,self.direction.z)
        start = o + carla.Location(0,0,0) #create new object by doing a mathematical operation
        end   = o +d
        start.x += t.location.x + b.location.x
        start.y += t.location.y + b.location.y
        start.z += t.location.z + b.extent.z
        end.x += t.location.x + b.location.x
        end.y += t.location.y + b.location.y
        end.z += t.location.z + b.extent.z

        world.world.debug.draw_line(carla.Vector3D(start),
                                    carla.Vector3D(end),
                                    thickness=0.1,
                                    color=carla.Color(5,0,0,20),life_time = 1)
        pass


cdef class Target:
    cdef public int idx
    cdef public BoundingBox box


    def __cinit__(self, int idx_, BoundingBox box_):
        self.idx = idx_
        self.box = box_


def rayBoxIntersection(ray, box):
    """
    Ray box intersection
    https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
    TODO: TEST
    """
    # inverse transform
    cdef Vector3D origin
    cdef Vector3D direction
    #TODO: implement Transform in cython as well,avoiding calls to carla api
    invRotation = carla.Transform(
        carla.Location(0,0,0),
        carla.Rotation(-box.rotation.pitch, -box.rotation.yaw, -box.rotation.roll))

    # transform ray
    # rotation and translation
    origin = ray.origin-box.location
    tmp = invRotation.transform(carla.Vector3D(origin.x, origin.y, origin.z))
    origin = Vector3D(tmp.x, tmp.y, tmp.z)
    tmp = invRotation.transform(carla.Vector3D(ray.direction.x, ray.direction.y, ray.direction.z))
    direction = Vector3D(tmp.x,tmp.y, tmp.z)

    #avoid division by zero:
    if (abs(direction.x) < 1e-6):
        direction.x = 1e-6 if direction.x > 0 else -1e-6
    if (abs(direction.y) < 1e-6):
        direction.y = 1e-6 if direction.y > 0 else -1e-6
    if (abs(direction.z) < 1e-6):
        direction.z = 1e-6 if direction.z > 0 else -1e-6

    # intersection with unit box [-extent.x,-extent.y,-extent.z],[extent.x,extent.y,extent.z]

    box_min = Vector3D(-box.extent.x,-box.extent.y,-box.extent.z)
    box_max = Vector3D( box.extent.x, box.extent.y, box.extent.z)

    tmin = (box_min - origin)/direction
    tmax = (box_max - origin)/direction

    t1 = Vector3D.min_(tmin,tmax)
    t2 = Vector3D.max_(tmin,tmax)
    tNear = t1.getMax()
    tFar = t2.getMin()
    does_intersect = tNear <= tFar and tNear >= 0 and tNear <= 1
    return does_intersect, tNear



def distanceAlongPath(waypoints: list, collisionBoxes, width, world = None, debug = False):
    cdef float travelDistance
    cdef int best_idx
    cdef int len_waypoints
    cdef Location box_location
    cdef Vector3D box_extent
    cdef Location waypoint
    cdef Location next_waypoint
    #cdef vector[Target] possibleTargets #doesn't work


    cdef float time
    cdef float timeLeft
    cdef float timeRight
    cdef float best_time


    travelDistance = 0.0
    best_idx = -1
    len_waypoints = len(waypoints)
    for i in range(len_waypoints - 1):
        if travelDistance > 110.0:
            return 110.0,best_idx

        waypoint = Location(waypoints[i].x,waypoints[i].y,waypoints[i].z)
        next_waypoint = Location(waypoints[i+1].x,waypoints[i+1].y,waypoints[i+1].z)
        #print(f"Segment length:{waypoint.distance(next_waypoint)}")
        if waypoint.distance(next_waypoint) < 1e-6:
            continue
        possibleTargets = []
        for idx,box in enumerate(collisionBoxes):
            box_location = Location(box.location.x, box.location.y, box.location.z)

            if (box_location.distance_sq(waypoint) < 100.0 or
                    box_location.distance_sq(next_waypoint) < 100.0):  # fast filtering
                box_extent = Vector3D(box.extent.x, box.extent.y, box.extent.z)
                box_rotation = Rotation(box.rotation.pitch, box.rotation.roll, box.rotation.yaw)
                possibleTargets.append(Target(idx,BoundingBox(box_location,box_extent, box_rotation)))

            if box.contains(waypoints[i], carla.Transform()):
                return travelDistance,idx #if waypoint inside bounding box == collision at t = 0


        if world and debug:
            ray = Ray(waypoint, next_waypoint)
            rayLeft, rayRight = calcCorners(ray, waypoint, next_waypoint, width)

            ray.draw(world)
            rayLeft.draw(world)
            rayRight.draw(world)

        if len(possibleTargets) == 0:  # no collision
            travelDistance += waypoint.distance(next_waypoint)
            continue
        #print(f"Debug:")
        #print(f"Segment length:{waypoint.distance(next_waypoint)}")
        #print(f"Waypoint 1:{waypoints[i]}")
        #print(f"Waypoint 1:{waypoint.x}, {waypoint.y}, {waypoint.z}")
        #print(f"Waypoint 2:{waypoints[i+1]}")
        #print(f"Waypoint 2:{next_waypoint.x}, {next_waypoint.y}, {next_waypoint.z}")
        ray = Ray(waypoint, next_waypoint)
        rayLeft,rayRight=calcCorners(ray,waypoint,next_waypoint, width)

        best_time = np.Inf
        flag = False
        for target in possibleTargets:
            idx = target.idx
            box = target.box

            collision, time = rayBoxIntersection(ray, box)
            collisionLeft,timeLeft=rayBoxIntersection(rayLeft, box)
            collisionRight, timeRight = rayBoxIntersection(rayRight, box)
            if collision and time < best_time:
                best_time = time
                best_idx = idx
                flag = True
            if collisionLeft and timeLeft < best_time:
                best_time = timeLeft
                best_idx = idx
                flag = True
            if collisionRight and timeRight < best_time:
                best_time = timeRight
                best_idx = idx
                flag = True

        # returned collision time should always be between 0 and 1
        if flag:  # collision found => return the total traveled length until this point
            travelDistance += waypoint.distance(next_waypoint) * best_time
            return travelDistance,best_idx
        else:
            travelDistance += waypoint.distance(next_waypoint)
    return travelDistance,best_idx

def calcCorners(middleRay,waypoint,nextwaypoint, width):
    v = Vector3D(middleRay.direction.y,-middleRay.direction.x,0)
    norm=np.sqrt(np.power(v.x,2)+np.power(v.y,2)+np.power(v.z,2))
    if (norm <= 1e-6):
        print(f"middleRay.x:{middleRay.direction.x}")
        print(f"middleRay.y:{middleRay.direction.y}")
        print(f"Norm{norm}")
        print('Error')
    assert(norm >= 1e-6)
    v.x = v.x/norm
    v.y = v.y/norm
    v.z = v.z/norm
    a=Location(waypoint.x+v.x*width,waypoint.y+v.y*width,waypoint.z)
    b = Location(nextwaypoint.x + v.x * width, nextwaypoint.y + v.y * width, nextwaypoint.z)
    c = Location(waypoint.x - v.x * width, waypoint.y - v.y * width, waypoint.z)
    d = Location(nextwaypoint.x - v.x * width, nextwaypoint.y - v.y * width, nextwaypoint.z)
    return Ray(a,b),Ray(c,d)