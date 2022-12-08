import carla
import numpy as np


class Ray(object):
    def __init__(self, start: carla.Location, end: carla.Location) -> None:
        super().__init__()
        self.origin = carla.Location(start.x,start.y,start.z)
        self.direction = end - start

    def draw(self,world):
        #world.debug.draw_line(self.origin, self.origin + self.direction)
        t = world.player.get_transform()
        b = world.player.bounding_box
        start = self.origin + carla.Location(0,0,0) #create new object by doing a mathematical operation
        end   = self.origin +self.direction
        start.x += t.location.x + b.location.x
        start.y += t.location.y + b.location.y
        start.z += t.location.z + b.location.z
        end.x += t.location.x + b.location.x
        end.y += t.location.y + b.location.y
        end.z += t.location.z + b.location.z

        world.world.debug.draw_line(carla.Vector3D(start),
                                    carla.Vector3D(end),
                                    thickness=0.1,
                                    color=carla.Color(5,0,0,20),life_time = 1)
        pass



def rayBoxIntersection(ray, box):
    """
    Ray box intersection
    https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
    TODO: TEST
    """
    # inverse transform

    invRotation = carla.Transform(
        carla.Location(0,0,0),
        carla.Rotation(-box.rotation.pitch, -box.rotation.yaw, -box.rotation.roll))

    # transform ray
    # rotation and translation
    origin = ray.origin-box.location
    origin = invRotation.transform(origin)
    direction = invRotation.transform(carla.Vector3D(ray.direction.x, ray.direction.y, ray.direction.z))
    #avoid division by zero:
    if (abs(direction.x) < 1e-6):
        direction.x = 1e-6 if direction.x > 0 else -1e-6
    if (abs(direction.y) < 1e-6):
        direction.y = 1e-6 if direction.y > 0 else -1e-6
    if (abs(direction.z) < 1e-6):
        direction.z = 1e-6 if direction.z > 0 else -1e-6

    # scaling => not needed, rest of algorithm takes care of this
    #origin.x /= box.extent.x
    #origin.y /= box.extent.y
    #origin.z /= box.extent.z
    #direction.x /= box.extent.x
    #direction.y /= box.extent.y
    #direction.z /= box.extent.z

    # intersection with unit box [-extent.x,-extent.y,-extent.z],[extent.x,extent.y,extent.z]

    box_min = carla.Vector3D(-box.extent.x,-box.extent.y,-box.extent.z)
    box_max = carla.Vector3D( box.extent.x, box.extent.y, box.extent.z)

    tmin = carla.Vector3D()
    tmax = carla.Vector3D()
    tmin.x = (box_min.x - origin.x) / direction.x
    tmin.y = (box_min.y - origin.y) / direction.y
    tmin.z = (box_min.z - origin.z) / direction.z

    tmax.x = (box_max.x - origin.x) / direction.x
    tmax.y = (box_max.y - origin.y) / direction.y
    tmax.z = (box_max.z - origin.z) / direction.z

    t1 = [min(tmin.x, tmax.x), min(tmin.y, tmax.y), min(tmin.z, tmax.z)]
    t2 = [max(tmin.x, tmax.x), max(tmin.y, tmax.y), max(tmin.z, tmax.z)]
    tNear = max(t1)
    tFar = min(t2)
    does_intersect = tNear <= tFar and tNear >= 0 and tNear <= 1
    return does_intersect, tNear


def distanceAlongPath(waypoints: list, collisionBoxes, width, world = None, debug = False):
    travelDistance = 0
    for i in range(len(waypoints) - 1):
        if (travelDistance > 400):
            return 400

        waypoint = waypoints[i]
        next_waypoint = waypoints[i + 1]
        if waypoint.distance(next_waypoint) < 1e-6:
            continue
        possibleTargets = []
        for box in collisionBoxes:
            if box.contains(waypoint, carla.Transform()):
                return travelDistance #if waypoint inside bounding box == collision at t = 0
            if (box.location.distance(waypoint) < 10 or
                    box.location.distance(next_waypoint) < 10):  # fast filtering
                possibleTargets.append(box)


        if world and debug:
            ray = Ray(waypoint, next_waypoint)
            rayLeft, rayRight = calcCorners(ray, waypoint, next_waypoint, width)

            ray.draw(world)
            rayLeft.draw(world)
            rayRight.draw(world)

        if len(possibleTargets) == 0:  # no collision
            travelDistance += waypoint.distance(next_waypoint)
            continue
        ray = Ray(waypoint, next_waypoint)
        rayLeft,rayRight=calcCorners(ray,waypoint,next_waypoint, width)

        best_time = np.Inf
        flag = False
        for target in possibleTargets:
            collision, time = rayBoxIntersection(ray, target)
            collisionLeft,timeLeft=rayBoxIntersection(rayLeft, target)
            collisionRight, timeRight = rayBoxIntersection(rayRight, target)
            if collision and time < best_time:
                best_time = time
                flag = True
            if collisionLeft and timeLeft < best_time:
                best_time = timeLeft
                flag = True
            if collisionRight and timeRight < best_time:
                best_time = timeRight
                flag = True

        # returned collision time should always be between 0 and 1
        if flag:  # collision found => return the total traveled length until this point
            travelDistance += waypoint.distance(next_waypoint) * best_time
            return travelDistance
        else:
            travelDistance += waypoint.distance(next_waypoint)
    return travelDistance

def calcCorners(middleRay,waypoint,nextwaypoint, width):
    v=carla.Vector3D(middleRay.direction.y,-middleRay.direction.x,0)
    norm=np.sqrt(np.power(v.x,2)+np.power(v.y,2)+np.power(v.z,2))
    if (norm <= 1e-6):
        print('Error')
    assert(norm > 1e-6)
    v.x = v.x/norm
    v.y = v.y/norm
    v.z = v.z/norm
    a=carla.Location(waypoint.x+v.x*width,waypoint.y+v.y*width,waypoint.z)
    b = carla.Location(nextwaypoint.x + v.x * width, nextwaypoint.y + v.y * width, nextwaypoint.z)
    c = carla.Location(waypoint.x - v.x * width, waypoint.y - v.y * width, waypoint.z)
    d = carla.Location(nextwaypoint.x - v.x * width, nextwaypoint.y - v.y * width, nextwaypoint.z)
    return Ray(a,b),Ray(c,d)