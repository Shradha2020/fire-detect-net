import numpy as np
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.registry import Registry

from .post_processing import get_panoptic_segmentation


#Understanding
 """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
 """
