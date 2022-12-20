import argparse
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import traceback

import carla
import easydict
from  easydict import EasyDict

from FinalIntegration.CarlaEnvironment.HUD import HUD
from FinalIntegration.Utils.Sensor import FollowCamera, Lidar, CollisionSensor, Camera
from FinalIntegration.Utils.CarlaAgent import  CarlaAgent
from FinalIntegration.CarlaEnvironment.CarlaWorld import CarlaWorld

def parse_args() ->EasyDict:
    parser = argparse.ArgumentParser(description='The Implementation using PyTorch')
    parser.add_argument('--host', type=str, default="127.0.0.1",
                        help='The host ip running carla. By default the localhost is used')
    parser.add_argument('--debug', '-d',action='store_true', help="Enable debug mode")

    config = EasyDict(vars(parser.parse_args()))
    config.fps = 20
    config.debug = False
    return config

def main(args):
    #create world
    carlaWorld = CarlaWorld(args)

    hud = HUD(1400, 700)
    carlaWorld.attachHUD(hud)
    try:
        carlaWorld.spawn(50, 0)
        for frame in range(100000):
            carlaWorld.step()
    except:
        traceback.print_exc()

    if carlaWorld:
        carlaWorld.destroy()


if __name__ == "__main__":
    args = parse_args()
    rl_config = {
        'model_path' : os.path.dirname(os.path.abspath(__file__)) +
                       "/FinalIntegration/models/ethereal-spaceship-160.pth",
        'history_frames': 3,
        'num_inputs': 12,  # =size of states!
        'num_actions': 101,
        'hidden': [128, 512, 512, 128, 64],
        'debug': False,
    }
    dl_lidar_config = {
        'K': 50,         #the number of top K
        'model_path': os.path.dirname(os.path.abspath(__file__)) + "/FinalIntegration/models/Utils_fpn_resnet_250.pth",
        'imagenet_pretrained': True,
        'head_conv': 64,
        'num_classes': 1,
        'num_center_offset': 2,
        'num_z': 1,
        'num_dim': 3,
        'num_direction': 2,  # sin, cos,
        'num_layers': 18,
    }
    dl_recognition_config = {
        'model_path':os.path.dirname(os.path.abspath(__file__)) + "/FinalIntegration/models/weights.pt",
        'hubconf_path':os.path.dirname(os.path.abspath(__file__)) + "/FinalIntegration/DeepLearningRecognition",
        'conf_threshold':0.25, #minimum confidence for object to be detected
        'iou_threshold':0.45,
        'max_detections':10,
    }

    args.rl_config = rl_config
    args.dl_lidar_config = dl_lidar_config
    args.dl_recognition_config = dl_recognition_config

    main(args)
