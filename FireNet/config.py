from detectron2.config import CfgNode as CN


def add_firenet_config(cfg):
    """
    Add config for firenet.
    """
    _C = cfg

    _C.MODEL.FIRENET = CN()

    # Number of branches for FIRENET.
    _C.MODEL.FIRENET.NUM_BRANCH = 3
    # Specify the dilations for each branch.
    _C.MODEL.FIRENET.BRANCH_DILATIONS = [1, 2, 3]
    # Specify the stage for applying FIRENET blocks. Default stage is Res4 according to the
    # TridentNet paper.
    _C.MODEL.FIRENET.TRIDENT_STAGE = "res4" # Need to check if needed.
    # Specify the test branch index FIRENET Fast inference:
    #   - use -1 to aggregate results of all branches during inference.
    #   - otherwise, only using specified branch for fast inference. Recommended setting is
    #     to use the middle branch.
    _C.MODEL.FIRENET.TEST_BRANCH_IDX = 1
