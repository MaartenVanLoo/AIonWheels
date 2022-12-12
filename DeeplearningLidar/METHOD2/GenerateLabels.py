import carla
import os
import glob
import numpy as np
class Labels:
    def __init__(self,folder_output):
        self.folder_output = folder_output
        self.label_output = folder_output + '/labels'
        os.makedirs(self.label_output) if not os.path.exists(self.label_output) else [os.remove(f) for f in glob.glob(self.label_output +"/*") if os.path.isfile(f)]

        # remove old test.txt file and create a new one
        os.remove(self.folder_output + '/test.txt') if os.path.isfile(self.folder_output + 'test.txt') else print('no test.txt folder found')
        f = open(self.folder_output+'/test.txt','w')
        f.close()

    def genLabels(self,world, KITTI, LIDAR, vehicles_list, pedestrian_list, idx, range=80, ):
        """
        generate labels by computing all labels for all non player vehicles in a given range[m].
        """
        truncated = 0
        occluded = 0
        alpha = 0

        sensor_loc = LIDAR.get_location()
        sensor_trans = LIDAR.get_transform()
        sensor_invtrans = sensor_trans.get_inverse_matrix()
        labels = []
        if vehicles_list:
            actors = world.get_actors(vehicles_list)

            for actor in actors:
                if actor.id == KITTI.id: # avoid detecting yourselves
                    continue
                vehicle_transform = actor.get_transform()
                if sensor_loc.distance(vehicle_transform.location) > range:
                    continue

                bbox = actor.bounding_box

                """transform bounding box from vehicle space to sensor space"""
                #vehicle space  to global space
                vehicle_transform.transform(bbox.location)
                bbox.rotation.pitch += vehicle_transform.rotation.pitch
                bbox.rotation.yaw += vehicle_transform.rotation.yaw
                bbox.rotation.roll += vehicle_transform.rotation.roll

                world.debug.draw_line(carla.Vector3D(vehicle_transform.location),
                                    carla.Vector3D(sensor_loc),
                                    thickness=0.1,
                                    color=carla.Color(5, 0, 0, 20), life_time=1)

                #global space to sensor space (all negative because transform is from sensor to global)
                bbox.location = applyTransform(sensor_invtrans,bbox.location)
                bbox.rotation.pitch -= sensor_trans.rotation.pitch
                bbox.rotation.yaw -= sensor_trans.rotation.yaw
                bbox.rotation.roll -= sensor_trans.rotation.roll

                """bbox is now in sensor space """

                labels.append({
                    'type': 'Car',
                    'xmin': 0,  # not needed for dataset
                    'ymin': 0,  # not needed for dataset
                    'xmax': 0,  # not needed for dataset
                    'ymax': 0,  # not needed for dataset
                    'h': bbox.extent.z * 2,
                    'w': bbox.extent.y * 2,
                    'l': bbox.extent.x * 2,
                    'x': bbox.location.x,
                    'y': bbox.location.y,
                    'z': bbox.location.z,
                    'yaw': np.radians(bbox.rotation.yaw),
                })

        if pedestrian_list:
            actors = world.get_actors(pedestrian_list)
            for actor in actors:
                if actor.id == KITTI.id: # avoid detecting yourselves
                    continue
                walker_transform = actor.get_transform()
                if sensor_loc.distance(walker_transform.location) > range:
                    continue

                bbox = actor.bounding_box

                """transform bounding box from vehicle space to sensor space"""
                # vehicle space  to global space
                walker_transform.transform(bbox.location)
                bbox.rotation.pitch += walker_transform.rotation.pitch
                bbox.rotation.yaw += walker_transform.rotation.yaw
                bbox.rotation.roll += walker_transform.rotation.roll

                world.debug.draw_line(carla.Vector3D(walker_transform.location),
                                      carla.Vector3D(sensor_loc),
                                      thickness=0.1,
                                      color=carla.Color(5, 0, 0, 20), life_time=1)

                # global space to sensor space
                bbox.location = applyTransform(sensor_invtrans,bbox.location)
                bbox.rotation.pitch -= sensor_trans.rotation.pitch
                bbox.rotation.yaw -= sensor_trans.rotation.yaw
                bbox.rotation.roll -= sensor_trans.rotation.roll

                """bbox is now in sensor space """

                labels.append({
                    'type': 'Pedestrian',
                    'xmin': 0,  # not needed for dataset
                    'ymin': 0,  # not needed for dataset
                    'xmax': 0,  # not needed for dataset
                    'ymax': 0,  # not needed for dataset
                    'h': bbox.extent.z * 2,
                    'w': bbox.extent.y * 2,
                    'l': bbox.extent.x * 2,
                    'x': bbox.location.x,
                    'y': bbox.location.y,
                    'z': bbox.location.z,
                    'yaw': np.radians(bbox.rotation.yaw),
                })

        with open(self.label_output + '/{:06d}.txt'.format(idx),'w') as f:
            for label in labels:
                f.write(f"{label['type']} {truncated} {occluded} {alpha} "
                        f"{label['xmin']} {label['ymin']} {label['xmax']} {label['ymax']} "
                        f"{label['h']} {label['w']} {label['l']} "
                        f"{label['x']} {label['y']} {label['z']} "
                        f"{label['yaw']}\n")
        with open(self.folder_output + '/test.txt','a') as f:
            f.write('{:06d}\n'.format(idx))
        pass

def applyTransform(transform, location):
    transform = np.array(transform)
    location = np.array([location.x, location.y, location.z ,1])
    location = location.dot(transform.T)
    return carla.Location(location[0],location[1], location[2])