import os
import sys

from .fpn_resnet import get_pose_net

def create_model(configs):
    """Create model based on architecture name"""
    num_layers = 18
    head_conv = 64
    heads = {
        'hm_cen': 1,
        'cen_offset': 2,
        'direction': 2,
        'z_coor': 1,
        'dim': 3
    }
    imagenet_pretrained = True

    print('using ResNet architecture with feature pyramid')
    model = get_pose_net(num_layers=num_layers, heads=heads, head_conv=head_conv,
                                        imagenet_pretrained=imagenet_pretrained)

    return model