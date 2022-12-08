import logging
import os

import carla
import math
import random
import time
import queue
import numpy as np
import cv2
from pascal_voc_writer import Writer


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


# Calculate 2D projection of 3D coordinate
def get_image_point(loc, K, w2c):
    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]


def generate_traffic(traffic_manager, client, blueprint_library, spawn_points, num_vehicles):
    """Generate traffic"""
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_random_device_seed(0)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.global_percentage_speed_difference(30.0)
    vehicle_bp = blueprint_library.filter('vehicle.*')
    spawn_points = spawn_points
    number_of_spawn_points = len(spawn_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    batch = []
    vehicles_list = []

    for n, transform in enumerate(spawn_points):
        if n >= num_vehicles:
            break
        blueprint = random.choice(vehicle_bp)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        # spawn
        # print("spawned")

        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, False):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    return vehicles_list


def generate_walkers(client, world, blueprint_library, spawn_points, number_of_walkers):
    return 0


def get_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


### Get numpy 2D array of vehicles' location and rotation from world reference, also locations from sensor reference
def get_list_transform(vehicles_list, sensor):
    t_list = []
    for vehicle in vehicles_list:
        v = vehicle.get_transform()
        transform = [v.location.x, v.location.y, v.location.z, v.rotation.roll, v.rotation.pitch, v.rotation.yaw]
        t_list.append(transform)
    t_list = np.array(t_list).reshape((len(t_list), 6))

    transform_h = np.concatenate((t_list[:, :3], np.ones((len(t_list), 1))), axis=1)
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    transform_s = np.dot(world_sensor_matrix, transform_h.T).T

    return t_list, transform_s


def filter_angle_occlusion(vehicles_list, world, vehicle, fov):
    filtered_vehicles = []
    for npc in world.get_actors().filter('vehicle*'):
        if npc.id in vehicles_list and npc.id != vehicle.id:
            # npc transform
            npc_transform = npc.get_transform()
            # vehicle transform
            vehicle_transform = vehicle.get_transform()
            # angle between npc and vehicle
            angle = np.arctan2(npc_transform.location, vehicle_transform.location) * 180 / np.pi
            selector = np.array(np.absolute(angle) < (int(fov) / 2))
            if selector:
                filtered_vehicles.append(npc)
    return filtered_vehicles


def main(town, num_of_vehicles, num_of_walkers, num_of_frames):
    # Simulator
    global xmin_bool
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.load_world(town)
    # world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    blueprint = blueprint_library.filter('model3')

    ego = world.spawn_actor(blueprint[0], random.choice(spawn_points))

    # spawn camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego)
    ego.set_autopilot(True)

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05  # (20fps)
    world.apply_settings(settings)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)


    # Get the attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)

    # Get the bounding boxes from traffic lights used later for red light detection
    bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)

    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    vehicles_list = generate_traffic(traffic_manager, client, blueprint_library, spawn_points, num_of_vehicles)
    # Spawn pedestrians and also detect the bounding boxes

    # Detect traffic Lights bounding boxes

    world.tick()
    image = image_queue.get()

    # Reshape the raw data into an RGB array
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    # Reshape pointcloud data

    # Display the image in an OpenCV display window
    cv2.namedWindow('CARLA RaceAI', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('CARLA RaceAI', img)
    cv2.waitKey(1)
    i = 50
    boxes = []
    try:
        ### Game loop ###
        while image.frame < num_of_frames:
            # Retrieve and reshape the image
            world.tick()
            image = image_queue.get()


            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # Get the camera matrix
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
            # only take measurements every 50 frames
            if image.frame % 30 == 0:
                i = 0
                # Save the image -- for export
                # Initialize the exporter

                boxes = []
                for npc in world.get_actors():
                    # Filter out the ego vehicle
                    if npc.id != ego.id and npc.id in vehicles_list:
                        bb = npc.bounding_box
                        dist = npc.get_transform().location.distance(ego.get_transform().location)

                        # Filter for the vehicles within 50m
                        if 0.5 < dist < 60:
                            forward_vec = ego.get_transform().get_forward_vector()
                            ray = npc.get_transform().location - ego.get_transform().location

                            if forward_vec.dot(ray) > 1:
                                p1 = get_image_point(bb.location, K, world_2_camera)
                                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                x_max = -10000
                                x_min = 10000
                                y_max = -10000
                                y_min = 10000

                                for vert in verts:
                                    p = get_image_point(vert, K, world_2_camera)
                                    # Find the rightmost vertex
                                    if p[0] > x_max:
                                        x_max = p[0]
                                    # Find the leftmost vertex
                                    if p[0] < x_min:
                                        x_min = p[0]
                                    # Find the highest vertex
                                    if p[1] > y_max:
                                        y_max = p[1]
                                    # Find the lowest vertex
                                    if p[1] < y_min:
                                        y_min = p[1]
                                name = npc.type_id.split('.')[2]
                                classification = 'car'
                                if name == 'ambulance' or name == 'fire_truck' or name == 'police' or name == 'police_2020':
                                    classification = 'emergency'
                                elif name == 'crossbike' or name == 'low_rider' or name == 'ninja' or name == 'zx125' or name == 'yzf' or name == 'omafiets':
                                    classification = 'motorcycle'
                                elif name == 'sprinter' or name == 'carlacola':
                                    classification = 'van'
                                    # Add the object to the frame (ensure it is inside the image)
                                if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                                    boxes.append([x_min, y_min, x_max, y_max, classification])

            i += 1
            if i == 3:
                # Compare the bounding boxes to every other bounding box
                # Filter out bad boxes
                for box in boxes:
                    for other_box in boxes:
                        # If the boxes are the same, skip
                        if box != other_box:
                            box_w = box[2] - box[0]
                            box_h = box[3] - box[1]
                            other_box_w = other_box[2] - other_box[0]
                            other_box_h = other_box[3] - other_box[1]
                            # Check if box is fully contained in other_box
                            if other_box[0] <= box[0] and other_box[1] <= box[1] and other_box[2] >= box[2] and \
                                    other_box[3] >= box[3]:
                                # If the box is fully contained, remove it
                                boxes.remove(box)
                                break

                image_path = 'output/camera_output/' + town + '/' + '%06d' % image.frame
                image.save_to_disk(image_path + '.png')
                writer = Writer(image_path + '.png', image_w, image_h)
                for box in boxes:
                    cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), (0, 0, 255, 255), 1)
                    cv2.line(img, (int(box[0]), int(box[3])), (int(box[2]), int(box[3])), (0, 0, 255, 255), 1)
                    cv2.line(img, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), (0, 0, 255, 255), 1)
                    cv2.line(img, (int(box[2]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255, 255), 1)

                    writer.addObject(box[4], box[0], box[1], box[2], box[3])

                    # Save the bounding boxes in the scene
                writer.save(image_path + '.xml')

                cv2.imshow('CARLA RaceAI', img)
                # save the image
                if not os.path.exists('output/camera_output/' + town + '/bbox'):
                    os.makedirs('output/camera_output/' + town + '/bbox/')
                cv2.imwrite('C:/Users/Bavo Lesy/PycharmProjects/RaceAI/output/camera_output/' + town + '/bbox/' + str(
                    image.frame) + '.png', img)

                if cv2.waitKey(1) == ord('q'):
                    break
    finally:
        # Destroy the actors
        for actor in world.get_actors().filter('vehicle.*'):
            actor.destroy()
        for actor in world.get_actors().filter('sensor.*'):
            actor.destroy()
        print('All actors destroyed.')

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Measurement every 50 frames, we want 400 measurement per town, so 30 * 400 = 16000+ 4000 for bad measurements
    # TO DO: change weather dynamically for each town

    frames = 20000
    num_vehicle = 75
    num_pedestrian = 30
    main('Town10HD', num_vehicle, num_pedestrian, frames)
    main('Town01', num_vehicle, num_pedestrian, frames)
    main('Town02', num_vehicle, num_pedestrian, frames)
    main('Town03', num_vehicle, num_pedestrian, frames)
    main('Town04', num_vehicle, num_pedestrian, frames)
    main('Town05', num_vehicle, num_pedestrian, frames)

