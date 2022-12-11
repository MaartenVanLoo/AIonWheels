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

        kit_pos = LIDAR.get_location()
        kit_trans = LIDAR.get_transform()
        inv_transform = carla.Transform(carla.Location(0,0,0), carla.Rotation(-kit_trans.rotation.pitch,-kit_trans.rotation.yaw,-kit_trans.rotation.roll))
        labels = []
        labels_old = []
        if vehicles_list:
            actors = world.get_actors(vehicles_list)

            for actor in actors:
                if actor.id == KITTI.id: # avoid detecting yourselves
                    continue
                transform = actor.get_transform()
                if kit_pos.distance(transform.location) > range:
                    continue

                bbox = actor.bounding_box
                point = inv_transform.transform(transform.location - kit_pos)

                labels.append({
                    'type': 'Car',
                    'xmin': 0,  # not needed for dataset
                    'ymin': 0,  # not needed for dataset
                    'xmax': 0,  # not needed for dataset
                    'ymax': 0,  # not needed for dataset
                    'h': bbox.extent.z * 2,
                    'w': bbox.extent.y * 2,
                    'l': bbox.extent.x * 2,
                    'x': bbox.location.x + point.x,
                    'y': bbox.location.y + point.y,
                    'z': bbox.location.z + point.z,
                    'yaw': np.radians(transform.rotation.yaw - kit_trans.rotation.yaw),
                })
                labels_old.append({
                    'type': 'Car',
                    'xmin': 0,  # not needed for dataset
                    'ymin': 0,  # not needed for dataset
                    'xmax': 0,  # not needed for dataset
                    'ymax': 0,  # not needed for dataset
                    'h': bbox.extent.z * 2,
                    'w': bbox.extent.y * 2,
                    'l': bbox.extent.x * 2,
                    'x': point.x,
                    'y': point.y,
                    'z': point.z,
                    'yaw': np.radians(transform.rotation.yaw - kit_trans.rotation.yaw),
                })


        if pedestrian_list:
            actors = world.get_actors(pedestrian_list)
            for actor in actors:
                if actor.id == KITTI.id: # avoid detecting yourselves
                    continue
                transform = actor.get_transform()
                if kit_pos.distance(transform.location) > range:
                    continue

                bbox = actor.bounding_box
                point = inv_transform.transform(transform.location - kit_pos)
                labels.append({
                    'type': 'Pedestrian',
                    'xmin': 0,  # not needed for dataset
                    'ymin': 0,  # not needed for dataset
                    'xmax': 0,  # not needed for dataset
                    'ymax': 0,  # not needed for dataset
                    'h': bbox.extent.z * 2,
                    'w': bbox.extent.y * 2,
                    'l': bbox.extent.x * 2,
                    'x': bbox.location.x + point.x,
                    'y': bbox.location.y + point.y,
                    'z': bbox.location.z + point.z,
                    'yaw': np.radians(transform.rotation.yaw - kit_trans.rotation.yaw),
                })
                labels_old.append({
                    'type': 'Pedestrian',
                    'xmin': 0,  # not needed for dataset
                    'ymin': 0,  # not needed for dataset
                    'xmax': 0,  # not needed for dataset
                    'ymax': 0,  # not needed for dataset
                    'h': bbox.extent.z * 2,
                    'w': bbox.extent.y * 2,
                    'l': bbox.extent.x * 2,
                    'x': point.x,
                    'y': point.y,
                    'z': point.z,
                    'yaw': np.radians(transform.rotation.yaw - kit_trans.rotation.yaw),
                })

        with open(self.label_output + '/{:06d}.txt'.format(idx),'w') as f:
            for label in labels:
                f.write(f"{label['type']} {truncated} {occluded} {alpha} "
                        f"{label['xmin']} {label['ymin']} {label['xmax']} {label['ymax']} "
                        f"{label['h']} {label['w']} {label['l']} "
                        f"{label['x']} {label['y']} {label['z']} "
                        f"{label['yaw']}\n")

        with open(self.label_output + '/{:06d}_old.txt'.format(idx),'w') as f:
            for label in labels_old:
                f.write(f"{label['type']} {truncated} {occluded} {alpha} "
                        f"{label['xmin']} {label['ymin']} {label['xmax']} {label['ymax']} "
                        f"{label['h']} {label['w']} {label['l']} "
                        f"{label['x']} {label['y']} {label['z']} "
                        f"{label['yaw']}\n")
        with open(self.folder_output + '/test.txt','a') as f:
            f.write('{:06d}\n'.format(idx))
        pass
