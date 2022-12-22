import math
import os

import cv2
import open3d as o3d
from kitti_dataset import KittiDataset

def o3d_boundingBox(label):
    #http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
    vertices = [
        [-1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1],
        [-1, 1, 1],
        [ 1,-1,-1],
        [ 1,-1, 1],
        [ 1, 1,-1],
        [ 1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]

    x = label[1]
    y = label[2]
    z = label[3]
    h = label[4]/2
    w = label[5]/2
    l = label[6]/2
    yaw = label[7]

    points = []
    points2 = []
    for vertex in vertices:
        v_x  = x + l * vertex[0]
        v_y  = y + w * vertex[1]
        v_z  = z + h * vertex[2]
        v_x, v_y = rotateZ(v_x,v_y, x, y, yaw)
        points.append([v_x,v_y,v_z])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def rotateZ(x, y , centerX, centerY, angle):
    # rotate acroding to yaw angle
    s = math.sin(angle)
    c = math.cos(angle)
    x = x - centerX
    y = y - centerY
    tmp_x = x * c - y * s
    tmp_y = x * s + y * c
    x = tmp_x + centerX
    y = tmp_y + centerY
    return x, y


def o3d_pointcloud(lidar):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar[:, 0:3])
    return pcd

def showCS():
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])
def show(lidar, labels):
    geometries = []
    for label in labels:
        geometries.append(o3d_boundingBox(label))
    geometries.append(o3d_pointcloud(lidar))
    geometries.append(showCS())
    o3d.visualization.draw_geometries(geometries)

if __name__ == '__main__':
    from easydict import EasyDict as edict

    configs = edict()
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.num_samples = None
    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.max_objects = 50
    configs.num_classes = 3
    configs.output_width = 500

    configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')
    # lidar_aug = OneOf([
    #     Random_Rotation(limit_angle=np.pi / 4, p=1.),
    #     Random_Scaling(scaling_range=(0.95, 1.05), p=1.),
    # ], p=1.)
    lidar_aug = None

    #dataset = KittiDataset(configs, mode='val', lidar_aug=lidar_aug, hflip_prob=0., num_samples=configs.num_samples)
    dataset = KittiDataset(configs, mode='test', lidar_aug=lidar_aug, hflip_prob=0., num_samples=configs.num_samples)

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')
    for idx in range(len(dataset)):
        lidar = dataset.get_lidar(idx)
        labels,_ = dataset.get_label(idx)

        #find min X, minY, min Z, max X, maxY, max Z
        min_value = lidar.min(axis =0)
        max_value = lidar.max(axis =0)
        print(f"Min:{min_value}")
        print(f"Max:{max_value}")



        #for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
        #    # Draw rotated box
        #    yaw = -yaw
        #    y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
        #    x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
        #    w1 = int(w / cnf.DISCRETIZATION)
        #    l1 = int(l / cnf.DISCRETIZATION)

        #https://towardsdatascience.com/guide-to-real-time-visualisation-of-massive-3d-point-clouds-in-python-ea6f00241ee0

        show(lidar, labels)
        if cv2.waitKey(0) & 0xff == 27:
            break

