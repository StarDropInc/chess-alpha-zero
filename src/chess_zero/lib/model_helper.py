from logging import getLogger

import os
import shutil
from datetime import datetime
from chess_zero.config import ResourceConfig
from chess_zero.lib.data_helper import get_newest_model_dirs

logger = getLogger(__name__)


def load_newest_model_weight(rc: ResourceConfig, model):
    """

    :param chess_zero.agent.model.ChessModel model:
    :return:
    """
    dirs = get_newest_model_dirs(rc)
    if not dirs:
        return False
    model_dir = dirs[-1]
    config_path = os.path.join(model_dir, rc.model_config_filename)
    weight_path = os.path.join(model_dir, rc.model_weight_filename)
    return model.load(config_path, weight_path)


def save_as_newest_model(rc: ResourceConfig, model):
    """

    :param chess_zero.agent.model.ChessModel model:
    :return:
    """
    model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    model_dir = os.path.join(rc.model_dir, rc.model_dirname_tmpl % model_id)
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, rc.model_config_filename)
    weight_path = os.path.join(model_dir, rc.model_weight_filename)
    model.save(config_path, weight_path)

def clear_old_models(rc: ResourceConfig):
    dirs = get_newest_model_dirs(rc)[:-1]
    for dir_ in dirs:
        if rc.keep_old_models:
            shutil.move(dir_, rc.old_model_dir)
        else:
            shutil.rmtree(dir_)
