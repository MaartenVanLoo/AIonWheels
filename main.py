import argparse
import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import traceback

import carla
import easydict
from easydict import EasyDict

from FinalIntegration.CarlaEnvironment.HUD import HUD
from FinalIntegration.Utils.Sensor import FollowCamera, Lidar, CollisionSensor, Camera
from FinalIntegration.Utils.CarlaAgent import CarlaAgent
from FinalIntegration.CarlaEnvironment.CarlaWorld import CarlaWorld

import numpy as np

def parse_args() ->EasyDict:
    parser = argparse.ArgumentParser(description='The Implementation using PyTorch')
    parser.add_argument('--host', type=str, default="127.0.0.1",
                        help='The host ip running carla. By default the localhost is used')
    parser.add_argument('--debug', '-d',action='store_true', help="Enable debug mode")
    parser.add_argument('--MT',action='store_true', help='Enable multithreading')

    parser.add_argument('--demo0', action='store_true', help='run demo 0')
    parser.add_argument('--demo1', action='store_true', help='run demo 1')
    parser.add_argument('--demo2', action='store_true', help='run demo 2')
    parser.add_argument('--demo3', action='store_true', help='run demo 3')

    config = EasyDict(vars(parser.parse_args()))
    config.fps = 20
    config.emergency_brake = 2
    config.forced_start = 10
    config.debug = False

    config.radar_debug = False
    config.crash_test = False


    return config

def main(args):
    # create world
    carlaWorld = CarlaWorld(args)

    hud = HUD(1600,700)
    carlaWorld.attachHUD(hud)

    print(f"Width : {carlaWorld.getPlayer().getWidth()}")
    print(f"Length: {carlaWorld.getPlayer().getLength()}")
    print(f"Height: {carlaWorld.getPlayer().getHeight()}")
    try:
        if args.demo0:
            carlaWorld.spawn(40,0)
        elif args.demo1:
            carlaWorld.spawn(70,0)
        else:
            carlaWorld.spawn(50,0)
        for frame in range(100000):
            carlaWorld.step()
    except:
        traceback.print_exc()

    if carlaWorld:
        carlaWorld.destroy()

def demo(args: EasyDict):
    #seed the random number generator according to the demos
    if args.demo0:
        import time
        #seed = int(time.time())
        #seed = 1674656090 #has firetruck? (Town01_Opt)

        seed = 1674680630 # "seems nice?" (Town01_Opt)
        print(f"Random seed:{seed}")
        np.random.seed(seed)
        pass
    elif args.demo1: #show different speeds
        seed = 16874102
        print(f"Random seed:{seed}")
        np.random.seed(seed)
        pass
    elif args.demo2:
        pass
    elif args.demo3:
        pass

if __name__ == "__main__":
    args = parse_args()
    rl_config = {
        'model_path': os.path.dirname(os.path.abspath(__file__)) +
                      "/FinalIntegration/models/ethereal-spaceship.pth",
        'history_frames': 3,
        'num_inputs': 12,  # =size of states!
        'num_actions': 101,
        'hidden': [128, 512, 512, 128, 64],
        'debug': False, #Doesn't do anything

        'target_speed': 50, # m/s

    }
    dl_lidar_config = {
        'K': 50,  # the number of top K
        'model_path': os.path.dirname(os.path.abspath(__file__)) +
                      "/FinalIntegration/models/1C_20OBj_Lidar.pth",
        'imagenet_pretrained': True,
        'head_conv': 64,
        'num_classes': 1,
        'num_center_offset': 2,
        'num_z': 1,
        'num_dim': 3,
        'num_direction': 2,  # sin, cos,
        'num_layers': 18,

        'debug':False,
    }
    dl_recognition_config = {
        'model_path':os.path.dirname(os.path.abspath(__file__)) + "/FinalIntegration/models/best21_01.pt",
        'hubconf_path':os.path.dirname(os.path.abspath(__file__)) + "/FinalIntegration/DeepLearningRecognition",
        'conf_threshold':0.45, #minimum confidence for object to be detected
        'iou_threshold':0.45,
        'max_detections':10,
    }

    args.rl_config = rl_config
    args.dl_lidar_config = dl_lidar_config
    args.dl_recognition_config = dl_recognition_config

    demo(args)
    main(args)
