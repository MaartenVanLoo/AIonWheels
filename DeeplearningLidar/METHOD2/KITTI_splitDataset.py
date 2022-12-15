import glob
from time import sleep

import numpy as np
import numpy.random
import os
import shutil as sh

from tqdm import tqdm


def split(count, p_train = 0.8, p_validation = 0.2):
    #make sure p_train + p_validation equals 1
    s = p_train + p_validation
    p_train /= s

    #random shuffel
    x = np.arange(count)
    np.random.shuffle(x)
    t_count = int(count * p_train)
    training, test = x[:t_count], x[t_count:]
    return training, test

def move(sample, frame_path, img_path, label_path, idx):
    sh.copy2(sample[1],frame_path + f"{idx:06d}.bin")
    sh.copy2(sample[2],img_path + f"{idx:06d}.jpg")
    sh.copy2(sample[3],label_path + f"{idx:06d}.txt")


def splitDataset(source_paths, target_path, p_train, p_validation):
    dataset = []
    currentDir = os.path.realpath(os.path.dirname(__file__))
    #reed index files
    print("Reading dataset files")
    for path in tqdm(source_paths):
        index_file = currentDir +"/"+ path + "test.txt"

        if not os.path.exists(index_file):
            print(f"Could not find index_file at {path}")
            continue
        sample_id_list = [int(x.strip()) for x in open(index_file).readlines()]
        print(f"\nReading {path}")
        for sample in tqdm(sample_id_list):
            frame_path = currentDir + "/" + path + f"frames/{sample:06d}.bin"
            img_path =   currentDir + "/" + path +  f"images_rgb/{sample:06d}.jpg"
            label_path = currentDir + "/" + path + f"labels/{sample:06d}.txt"
            if not os.path.exists(frame_path):
                print(f"Could not find frame: {frame_path}")
                continue
            if not os.path.exists(img_path):
                print(f"Could not find image path: {img_path}")
                continue
            if not os.path.exists(label_path):
                print(f"Could not find label path: {label_path}")
                continue
            dataset.append([sample,frame_path, img_path, label_path,path])
    #get count:
    total_datapoints =len(dataset);
    print(f"Found {total_datapoints} datapoints")
    if not dataset:
        return
    #get split indeces
    train_id, valid_id = split(total_datapoints, p_train, p_validation)

    #write imageSets
    sleep(0.1)
    print("Writing set configurations")
    set_path = currentDir + "/" + target_path + "ImageSets/"
    if not os.path.exists(set_path):
        os.makedirs(set_path)
    with open(set_path+"train.txt","w") as f:
        for id in np.sort(train_id):
            f.write(f"{id:06d}\n")
    with open(set_path+"val.txt","w") as f:
        for id in np.sort(valid_id):
            f.write(f"{id:06d}\n")

    #move files
    training_path = currentDir + "/" + target_path + "training/"
    velodyne_path = training_path + "velodyne/"
    image_path = training_path + "image_2/"
    label_path = training_path + "label_2/"
    for path in [training_path,velodyne_path, image_path, label_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            [os.remove(f) for f in glob.glob(path + "/*") if os.path.isfile(f)]

    sleep(0.1)
    print("\nWriting traceability paths")
    with open(currentDir+"/"+ target_path + "traceability.txt","w") as f:
        f.write(f"new_id\tpath\tid\n")
        for idx,s in enumerate(tqdm(dataset)):
            f.write(f"{idx:06d}\t{s[0]:06d}\t{s[-1]}\n")

    sleep(0.1)
    print("\nCreating dataset")
    for idx,sample in enumerate(tqdm(dataset)):
        move(sample,velodyne_path, image_path, label_path,idx)



if __name__ == "__main__":
    config = {
        'target_path':"dataset/kitti/",
        'source_paths':[
            "KITTI_Dataset_Carla_v0.9.13/carla/Maps/Town01/",
            "KITTI_Dataset_Carla_v0.9.13/carla/Maps/Town02/",
            "KITTI_Dataset_Carla_v0.9.13/carla/Maps/Town03/",
            "KITTI_Dataset_Carla_v0.9.13/carla/Maps/Town04/",
            "KITTI_Dataset_Carla_v0.9.13/carla/Maps/Town05/",
            "KITTI_Dataset_Carla_v0.9.13/carla/Maps/Town06/",
            "KITTI_Dataset_Carla_v0.9.13/carla/Maps/Town07/",
            "KITTI_Dataset_Carla_v0.9.13/carla/Maps/Town10HD/",
        ]

    }
    splitDataset(config.get("source_paths"), config.get("target_path"), 0.8,0.2)