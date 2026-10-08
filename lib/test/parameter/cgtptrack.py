from copy import deepcopy
from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.cgtptrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/cgtptrack/%s.yaml' % yaml_name)
    config = deepcopy(cfg)
    update_config_from_file(yaml_file, base_cfg=config)
    params.cfg = config
    print("test config: ", config)

    # template and search region
    params.template_factor = config.TEST.TEMPLATE_FACTOR
    params.template_size = config.TEST.TEMPLATE_SIZE
    params.search_factor = config.TEST.SEARCH_FACTOR
    params.search_size = config.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/cgtptrack/%s/CGTPTrack_ep%04d.pth.tar" %
                                     (yaml_name, config.TEST.EPOCH))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
