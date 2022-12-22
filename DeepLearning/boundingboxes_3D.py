import carla
import random
import queue
import numpy as np
import cv2
from pascal_voc_writer import Writer
#import trafficgenerator
import logging

### Geometric transformations ###
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

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def tarffic_generator(traffic_manager, client, bp_lib, spawn_points, amount_of_vehicles):
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_random_device_seed(0)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.global_percentage_speed_difference(30.0)
    vehicle_bp = bp_lib.filter('vehicle.*')
    spawn_points = spawn_points
    number_of_spawn_points = len(spawn_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    batch = []
    vehicles_list = []

    for n, transform in enumerate(spawn_points):
        if n >= amount_of_vehicles:
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

def main(map, amount_of_vehicles):
    #  Set up simulator #
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world(map) # Define the map to be used
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points() # Get the map spawn points

    #  Set up ego car   #
    ego_car_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
    ego_car = world.try_spawn_actor(ego_car_bp, random.choice(spawn_points))
    ego_car.set_autopilot(True)

    #  Set up camera  #
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('sensor_tick', '0')
    cam_bp.set_attribute('shutter_speed', '200')
    cam_bp.set_attribute('bloom_intensity', '0.675')
    cam_bp.set_attribute('gamma', '2.2')
    cam_bp.set_attribute('lens_flare_intensity', '0.1')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    cam_bp.set_attribute("image_size_x",str(600))
    cam_bp.set_attribute("image_size_y",str(400))
    cam_bp.set_attribute("fov",str(110))
    camera = world.spawn_actor(cam_bp, camera_init_trans, attach_to=ego_car, attachment_type=carla.AttachmentType.Rigid) #geen rigid

    # Get the attributes from the camera
    image_w = cam_bp.get_attribute("image_size_x").as_int()
    image_h = cam_bp.get_attribute("image_size_y").as_int()
    fov = cam_bp.get_attribute("fov").as_float()

    #  Set up settings #
    settings = world.get_settings() # Get the current settings
    settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05# Sets the fixed time step 20 FPS
    world.apply_settings(settings)

    #  Set up image queue to retrieve camera data #
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)

    #  Bounding boxes   #
    #TO BE ADDED: city lights, traffic lights, traffic signs, pedestrians, vehicles, etc.

    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

    traffic_manager = client.get_trafficmanager()
    vehicle_list = tarffic_generator(traffic_manager, client, bp_lib, spawn_points, amount_of_vehicles)

    #  Set up lights #
    #traffic_lights = world.get_actors().filter('*light*')
    #for lights in traffic_lights:
    #    if isinstance(lights, carla.TrafficLight):
    #        # for any light, first set the light state, then set time. for yellow it is
    #        # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
    #        lights.set_state(carla.TrafficLightState.Green)
    #        lights.set_green_time(1000.0)
            # actor_.set_green_time(5000.0)
            # actor_.set_yellow_time(1000.0)


    # Display the image in an OpenCV display window
    world.tick()
    image = image_queue.get()
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    cv2.namedWindow('BoundingBoxes', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('BoundingBoxes', img)
    cv2.waitKey(1)

    try:
        ### Game loop ###
        while True:

            # Retrieve and reshape the image
            world.tick()
            image = image_queue.get()
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # Get the camera matrix
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # Take snap every x frames
            if image.frame % 20 == 0:

                #geen boxes
                for npc in world.get_actors(): #* * matches everything different
                    # Filter out the ego vehicle
                    if npc.id != ego_car.id and npc.id in vehicle_list: #and npc.bounding_box.compute_visibility(npc.bounding_box.get_transform(), world) :

                        bb = npc.bounding_box
                        dist = npc.get_transform().location.distance(ego_car.get_transform().location)
                        print(npc.bounding_box.get_transform())

                        # Filter for the vehicles within 50m
                        if 2 < dist < 50:
                            forward_vec = ego_car.get_transform().get_forward_vector()
                            ray = npc.get_transform().location - ego_car.get_transform().location

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
                                    # Find the lowest  vertex
                                    if p[1] < y_min:
                                        y_min = p[1]

                                #image = cv2.line(image, start_point, end_point, color, thickness)
                                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)),(0, 255, 0), 1)
                                #cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                                #cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                                #cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                                #cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)

                                #classification for xml file
                                type = npc.type_id.split('.')[2]
                                classification_text = ''
                                # Write case type statement not possible in 3.7 LOL
                                # Used to see what types of vehicles there are: https://www.youtube.com/watch?v=kL55VnuDpxw
                                if type == 'crossbike' or type == 'omafiets' or type == 'century':
                                    classification_text = 'bike'
                                elif type == 'low_rider' or type == 'zx125' or type == 'yzf' or type == 'ninja':
                                    classification_text = 'motorcycle'
                                elif type == 'firetruck':
                                    classification_text = 'firetruck'
                                elif type == 'charger_police' or type == 'charger_police_2020':
                                    classification_text = 'police'
                                elif type == 'ambulance':
                                    classification_text = 'ambulance'
                                elif type == 'carlacola':
                                    classification_text = 'truck'
                                else:
                                    classification_text = 'car'

                                #Put label next to image
                                img = cv2.putText(img, text= classification_text, org=(int(x_min), int(y_min)),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 0, 255, 255),thickness=1)

                # Save the image -- for export
                frame_path = 'output/%06d' % image.frame
                image.save_to_disk(frame_path + '.png')

                # Initialize the exporter
                writer = Writer(frame_path + '.png', image_w, image_h)

                # Add the object to the frame (ensure it is inside the image)
                if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                    writer.addObject(classification_text, x_min, y_min, x_max, y_max)

                # Save the bounding boxes in the scene
                writer.save(frame_path + '.xml')

                # Display the image in an OpenCV display window
                # add lines to save image
                cv2.imshow('BoundingBoxes',img)
                cv2.imwrite('output/%06d.png' % image.frame, img)
                if cv2.waitKey(1) == ord('q'):
                    break

    except:
        pass

    finally:
        if camera is not None:
            camera.destroy()
        if vehicle_list:
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Measurement every 50 frames, we want 400 measurement per town, so 30 * 400 = 16000+ 4000 for bad measurements
    # TO DO: change weather dynamically for each town

    main('Town03', amount_of_vehicles = 50)
