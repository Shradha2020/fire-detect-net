from detectron2.config import CfgNode as CN

from detectron2.config import CfgNode as CN
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config



def add_firenet_config(cfg):
    add_panoptic_deeplab_config(cfg)
