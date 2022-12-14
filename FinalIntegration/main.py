import argparse
import os
import traceback

from  easydict import EasyDict

from CarlaEnvironment.HUD import HUD
from Generator.Sensor import FollowCamera, Lidar, CollisionSensor, Camera
from Generator.CarlaAgent import  CarlaAgent
from CarlaEnvironment.CarlaWorld import CarlaWorld

from ReinforcementLearning.ReinforcementLearning import ReinforcementLearning
from DeepLearningLidar.DeepLearningLidar import DeeplearningLidar
from DeepLearningRecognition.DeepLearningRecognition import DeepLearningRecognition

def parse_args() ->EasyDict:
    parser = argparse.ArgumentParser(description='The Implementation using PyTorch')
    parser.add_argument('--host', type=str, default="127.0.0.1",
                        help='The host ip running carla. By default the localhost is used')
    config = EasyDict(vars(parser.parse_args()))
    config.fps = 20
    return config

def main(rl_config,args):
    carlaWorld = CarlaWorld(args)


    # AI modules:
    rl_module = ReinforcementLearning(carlaWorld,rl_config)
    dl_lidar = DeeplearningLidar()
    dl_recognition = DeepLearningRecognition()
    hud = HUD(1000,500)
    carlaWorld.attachHUD(hud)
    try:
        carlaWorld.spawn(50,0)
        for frame in range(10000):
            carlaWorld.step(rl_module, dl_lidar, dl_recognition)
    except:
        traceback.print_exc()

    if carlaWorld:
        carlaWorld.destroy()


if __name__ == "__main__":
    args = parse_args()
    rl_config = {
        "model_path" : os.path.dirname(os.path.abspath(__file__)) + "/models/pious-blaze-154.pth",
        'history_frames': 3,
        'num_inputs': 12,  # =size of states!
        'num_actions': 101,
        'hidden': [128, 512, 512, 128, 64],
        'debug': False,
    }
    main(rl_config, args)
