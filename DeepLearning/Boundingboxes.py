import carla
import random
import queue
import numpy as np
import cv2
from pascal_voc_writer import Writer
import trafficgenerator

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

# ----------------- #
#  Set up simulator #
# ----------------- #
client = carla.Client('localhost', 2000)
map = 'Town02'
world = client.load_world(map) # Define the map to be used
world = client.get_world()

spawn_points = world.get_map().get_spawn_points() # Get the map spawn points

# ----------------- #
#  Set up ego car   #
# ----------------- #
ego_car_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
ego_car = world.try_spawn_actor(ego_car_bp, random.choice(spawn_points))
ego_car.set_autopilot(True)

# ----------------- #
#  Set up sensors   #
# ----------------- #
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(cam_bp, camera_init_trans, attach_to=ego_car)

image_queue = queue.Queue() # Create a queue to store and retrieve the sensor data
camera.listen(image_queue.put)

# Get the attributes from the camera
image_w = cam_bp.get_attribute("image_size_x").as_int()
image_h = cam_bp.get_attribute("image_size_y").as_int()
fov = cam_bp.get_attribute("fov").as_float()

settings = world.get_settings() # Get the current settings
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.1 # Sets the fixed time step 20 FPS
world.apply_settings(settings)

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)

# ----------------- #
#  Bounding boxes   #
# ----------------- #

#TO BE ADDED: city lights, traffic lights, traffic signs, pedestrians, vehicles, etc.

# Remember the edge pairs
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

traffic_manager = client.get_trafficmanager()
traffic_list = trafficgenerator.generateTraffic(world, client, traffic_manager, 20, args={})

list_actor = world.get_actors()
for actor_ in list_actor:
    if isinstance(actor_, carla.TrafficLight):
        # for any light, first set the light state, then set time. for yellow it is
        # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
        actor_.set_state(carla.TrafficLightState.Green)
        actor_.set_green_time(1000.0)
        # actor_.set_green_time(5000.0)
        # actor_.set_yellow_time(1000.0)

# Retrieve the first image
world.tick()
image = image_queue.get()

# Reshape the raw data into an RGB array
img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

# Display the image in an OpenCV display window
cv2.namedWindow('BoundingBoxes', cv2.WINDOW_AUTOSIZE)
cv2.imshow('BoundingBoxes',img)
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
        #if image.frame % 10 == 0:
        # Save the image -- for export
        frame_path = 'output/%06d' % image.frame
        image.save_to_disk(frame_path + '.png')

        # Initialize the exporter
        writer = Writer(frame_path + '.png', image_w, image_h)

        for npc in world.get_actors().filter('*vehicle*'): #* * matches everything different

            # Filter out the ego vehicle
            if npc.id in traffic_list:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(ego_car.get_transform().location)

                # Filter for the vehicles within 50m
                if 1 < dist < 40:
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
                        img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                        #cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                        #cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                        #cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                        #cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)

                        #Put label next to image
                        img = cv2.putText(img, text = npc.type_id.split('.')[1], org = (int(x_min), int(y_min)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.3, color = (0,0,255, 255), thickness = 1)

                        # Add the object to the frame (ensure it is inside the image)
                        if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                                writer.addObject(npc.type_id.split('.')[1], x_min, y_min, x_max, y_max)

        # Save the bounding boxes in the scene
        writer.save(frame_path + '.xml')

        # Display the image in an OpenCV display window
        # add lines to save image
        cv2.imshow('BoundingBoxes',img)
        cv2.imwrite('output/%06d.png' % image.frame, img)
        if cv2.waitKey(1) == ord('q'):
            break

except: pass
finally:
    if camera is not None:
        camera.destroy()
    if traffic_list:
        client.apply_batch([carla.command.DestroyActor(x) for x in traffic_list])
    cv2.destroyAllWindows()
