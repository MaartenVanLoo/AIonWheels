import argparse
import logging
import os
import traceback

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

    config = EasyDict(vars(parser.parse_args()))
    config.fps = 20
    config.debug = False
    return config

def main(args):
    #create world
    carlaWorld = CarlaWorld(args)

    hud = HUD(1400,700)
    carlaWorld.attachHUD(hud)
    try:
        carlaWorld.spawn(50,0)
        for frame in range(100000):
            carlaWorld.step()
    except:
        traceback.print_exc()

    if carlaWorld:
        carlaWorld.destroy()


if __name__ == "__main__":
    args = parse_args()
    rl_config = {
        "model_path" : os.path.dirname(os.path.abspath(__file__)) + "/FinalIntegration/models/pious-blaze-154.pth",
        'history_frames': 3,
        'num_inputs': 12,  # =size of states!
        'num_actions': 101,
        'hidden': [128, 512, 512, 128, 64],
        'debug': False,
    }
    dl_lidar_config = {
        'K': 50 #the number of top K
    }
    dl_recognition_config = {

    }

    args.rl_config = rl_config
    args.dl_lidar_config = dl_lidar_config
    args.dl_recognition_config = dl_recognition_config

    main(args)
