import os

import cv2
from tqdm import tqdm


def png_to_jpg(folder, in_file, out_file, quality=95):
    if not os.path.exists(folder + in_file):
        return
    image = cv2.imread(folder + in_file)
    cv2.imwrite(folder + out_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


if __name__ == "__main__":
    #path = f"D:/Documenten/Universiteit/Master EI/I-DistributedAI/Video's/Lidar_stopmotion/"
    #for n in tqdm(range(6000)):
    #    png_to_jpg(path, '{:d}_lidar.png'.format(n), '{:d}_lidar.jpg'.format(n), 80)
    #quit(0)
    maps = ['Town01', 'Town02','Town03','Town04','Town05','Town06','Town07','Town10HD']
    for map in maps:
        path_rgb = f"KITTI_Dataset_CARLA_v0.9.13/Carla/Maps/{map}/generated/images_rgb/"
        path_ss = f"KITTI_Dataset_CARLA_v0.9.13/Carla/Maps/{map}/generated/images_ss/"

        for n in tqdm(range(2000)):
            png_to_jpg(path_rgb, '{:06d}.png'.format(n), '{:06d}.jpg'.format(n), 80)
            # png_to_jpg(path_ss, '{:06d}.png'.format(n),'{:06d}.jpg'.format(n),80)
