import math
import queue
from torch.multiprocessing import Queue as mpQueue
import weakref

import carla
import numpy as np


class Sensor(object):
    def __init__(self, parent: carla.Actor, world:carla.World, transform : carla.Transform) -> None:
        super().__init__()
        self._queue = queue.PriorityQueue()
        self.parent = parent
        self.world = world
        self.transform = transform
        self.sensor = None
        self.state = None

    @staticmethod
    def sensor_callback(frame, timestamp,sensor_transform , data, sensor_queue):
        sensor_queue.put((frame,data))

    def step(self):
        pass

    def getState(self):
        return self.state

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()



class Lidar(Sensor):
    def __init__(self,parent: carla.Actor, world:carla.World, transform : carla.Transform) -> None:
        super().__init__(parent, world,transform)
        self.bp = self._getBlueprint()
        self.sensor = world.spawn_actor(self.bp, self.transform, attach_to=self.parent)
        self.sensor.listen(
            lambda data: Sensor.sensor_callback(data.frame, data.timestamp,data.transform, data, self._queue))

        self._all_points = np.empty(shape=(0), dtype=np.float32)

        #packet counter
        self.i_packet = 0
        settings = world.get_settings()
        self.packet_per_frame = 1 / (
                    self.bp.get_attribute('rotation_frequency').as_float() * settings.fixed_delta_seconds)


    def step(self):
        while not self._queue.empty():
            frame, data = self._queue.get()
            print(data)
            buffer = np.frombuffer(data.raw_data, dtype=np.float32)
            self._all_points = np.concatenate((self._all_points, buffer), axis=0)

            self.i_packet += 1
            if self.i_packet >= self.packet_per_frame:
                self.i_packet = 0
                #when frame is finished, update state:
                self.state = self._all_points.reshape(-1, 4) #Todo
                self._all_points = np.empty(shape=(0), dtype=np.float32)

    def _getBlueprint(self):
        library = self.world.get_blueprint_library()
        lidar_bp = library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', '90.0')  # 80.0 m
        lidar_bp.set_attribute('points_per_second', str(1800000))
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', str(3))
        lidar_bp.set_attribute('lower_fov', str(-24.8))
        return lidar_bp

class AsyncLidar(Sensor):
    def __init__(self,parent: carla.Actor, world:carla.World, transform : carla.Transform) -> None:
        super().__init__(parent, world,transform)
        self.bp = self._getBlueprint()
        self.sensor = world.spawn_actor(self.bp, self.transform, attach_to=self.parent)
        self.sensor.listen(
            lambda data: self.sensor_callback(data.frame, data.timestamp,data.transform, data, self._queue))

        self._all_points = np.empty(shape=(0), dtype=np.float32)

        #packet counter
        self.i_packet = 0
        settings = world.get_settings()
        self.packet_per_frame = 1 / (
                self.bp.get_attribute('rotation_frequency').as_float() * settings.fixed_delta_seconds)

        self._output_queue = queue.Queue()

    def getQueue(self):
        return self._queue

    def sensor_callback(self,frame, timestamp, sensor_transform, data, sensor_queue):
        print(data)
        buffer = np.frombuffer(data.raw_data, dtype=np.float32)
        self._all_points = np.concatenate((self._all_points, buffer), axis=0)

        self.i_packet += 1
        if self.i_packet >= self.packet_per_frame:
            self.i_packet = 0
            #when frame is finished, update state:
            self._queue.put((frame, self._all_points.reshape(-1, 4)))
            self._all_points = np.empty(shape=(0), dtype=np.float32)

    def step(self):
        pass

    def _getBlueprint(self):
        library = self.world.get_blueprint_library()
        lidar_bp = library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', '90.0')  # 80.0 m
        lidar_bp.set_attribute('points_per_second', str(1800000))
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', str(3))
        lidar_bp.set_attribute('lower_fov', str(-24.8))
        return lidar_bp



class Camera(Sensor):
    def __init__(self,parent: carla.Actor,world:carla.World, transform : carla.Transform) -> None:
        super().__init__(parent,world,transform)
        self.bp = self._getBlueprint()
        self.sensor = world.spawn_actor(self.bp, self.transform, attach_to=self.parent)
        self.sensor.listen(
            lambda data: Sensor.sensor_callback(data.frame, data.timestamp, data.transform, data, self._queue))

    def step(self):
        while not self._queue.empty():
            frame, image = self._queue.get()
            print(image)
            #only when no more images are available, last image is current state
            if self._queue.empty():
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.state = array

    def _getBlueprint(self):
        library = self.world.get_blueprint_library()
        camera_bp = library.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '640')
        camera_bp.set_attribute('fov', '100')  # 72 degrees # Always fov on width even if width is different than height
        camera_bp.set_attribute('sensor_tick', str(1/20))  # 20Hz camera
        camera_bp.set_attribute('gamma', '2.2')
        #camera_bp.set_attribute('motion_blur_intensity', '0')
        #camera_bp.set_attribute('motion_blur_max_distortion', '0')
        #camera_bp.set_attribute('motion_blur_min_object_screen_size', '0')
        camera_bp.set_attribute('shutter_speed', '200')  # 1 ms shutter_speed
        camera_bp.set_attribute('bloom_intensity', '0.675')
        camera_bp.set_attribute('lens_flare_intensity', '0.675')
        #camera_bp.set_attribute('lens_k', '0')
        #camera_bp.set_attribute('lens_kcube', '0')
        #camera_bp.set_attribute('lens_x_size', '0')
        #camera_bp.set_attribute('lens_y_size', '0')
        return camera_bp



class AsyncCamera(Sensor):

    def __init__(self, parent: carla.Actor, world: carla.World, transform: carla.Transform) -> None:
        super().__init__(parent, world, transform)
        self.bp = self._getBlueprint()
        self.sensor = world.spawn_actor(self.bp, self.transform, attach_to=self.parent)
        self.sensor.listen(
            lambda data: AsyncCamera.sensor_callback(data.frame, data.timestamp, data.transform, data, self._queue))

    @staticmethod
    def sensor_callback(frame, timestamp, sensor_transform, data, sensor_queue):
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        sensor_queue.put((frame, array))
        #sensor_queue.put

    def _getBlueprint(self):
        library = self.world.get_blueprint_library()
        camera_bp = library.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '640')
        camera_bp.set_attribute('fov', '100')  # 72 degrees # Always fov on width even if width is different than height
        camera_bp.set_attribute('sensor_tick', str(1 / 20))  # 20Hz camera
        camera_bp.set_attribute('gamma', '2.2')
        camera_bp.set_attribute('bloom_intensity', '0.675')
        camera_bp.set_attribute('lens_flare_intensity', '0.675')
        return camera_bp

    def step(self):
        pass

    def getQueue(self):
        return self._queue


class FollowCamera(Sensor):
    def __init__(self, parent: carla.Actor,world:carla.World) -> None:
        super().__init__(parent, world,carla.Transform())

    def step(self):
        self.transform = self.parent.get_transform()
        rot = self.transform.rotation
        rot.pitch = -25
        self.world.get_spectator().set_transform(
            carla.Transform(self.transform.transform(carla.Location(x=-15, y=0, z=5)), rot))


class CollisionSensor(Sensor):
    def __init__(self, parent: carla.Actor, world: carla.World) -> None:
        super().__init__(parent, world, carla.Transform())
        self.blueprint = self._getBlueprint()
        self.sensor = world.spawn_actor(self.blueprint, self.transform, attach_to=self.parent)
        self.history = []

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))


    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = self._get_actor_display_name(event.other_actor)
        #self.hud.notification('Collision with %r' % actor_type)
        print(f'Collision with {actor_type}')
        self.state = (event.frame, f'Collision with {actor_type}')
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

    def _getBlueprint(self):
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        return blueprint

    @staticmethod
    def _get_actor_display_name(actor, truncate=250):
        """Method to get actor display name"""
        name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
        return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name



