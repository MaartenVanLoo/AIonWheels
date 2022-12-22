import os
import sys

from .fpn_resnet import get_pose_net

def create_model(config):
    """Create model based on architecture name"""
    heads = {
        'hm_cen': config.get('num_classes'),
        'cen_offset': config.get('num_center_offset'),
        'direction': config.get('num_direction'),
        'z_coor': config.get('num_z'),
        'dim': config.get('num_dim')
    }

    print('using ResNet architecture with feature pyramid')

    model = get_pose_net(num_layers=config.get('num_layers'), heads=heads, head_conv=config.get('head_conv'),
                                        imagenet_pretrained=config.get('imagenet_pretrained'), config=config)

    return model