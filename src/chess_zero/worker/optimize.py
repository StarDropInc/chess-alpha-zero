from logging import getLogger
from time import sleep
from time import time

import keras.backend as k
import numpy as np
import os
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

from chess_zero.agent.model_chess import ChessModel, objective_function_for_policy, objective_function_for_value
from chess_zero.config import Config
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from chess_zero.lib.model_helper import load_newest_model_weight, save_as_newest_model, clear_old_models
from chess_zero.env.chess_env import ChessEnv
import chess

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.5)
    return OptimizeWorker(config).start()


class OptimizeWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ChessModel
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.dataset = None
        self.optimizer = None

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        tc = self.config.trainer
        last_load_data_step = last_save_step = total_steps = tc.start_total_steps
        min_data_size_to_learn = tc.min_data_size_to_learn
        self.load_play_data()

        while True:
            if self.dataset_size < min_data_size_to_learn:
                logger.info(f"dataset_size={self.dataset_size} is less than {min_data_size_to_learn}")
                sleep(60)
                self.load_play_data()
                continue
            self.update_learning_rate(total_steps)
            steps = self.train_epoch(tc.epoch_to_checkpoint)
            total_steps += steps
            if last_save_step + tc.save_model_steps < total_steps:
                self.replace_current_model()
                last_save_step = total_steps

            if last_load_data_step + tc.load_data_steps < total_steps:
                self.load_play_data()
                last_load_data_step = total_steps

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.dataset
        tensorboard = TensorBoard(log_dir=os.path.join(self.config.resource.log_dir, str(time())), histogram_freq=32, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        self.model.model.fit(state_ary, [policy_ary, value_ary], batch_size=tc.batch_size, epochs=epochs, callbacks=[tensorboard])
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        self.optimizer = SGD(lr=1e-2, momentum=0.9)
        losses = [objective_function_for_policy, objective_function_for_value]
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def update_learning_rate(self, total_steps):
        # see "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
        if total_steps < 100000:
            lr = 2e-1
        elif total_steps < 500000:
            lr = 2e-2
        elif total_steps < 900000:
            lr = 2e-3
        else:
            lr = 2e-4
        k.set_value(self.optimizer.lr, lr)
        logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def replace_current_model(self):
        save_as_newest_model(self.config.resource, self.model)
        clear_old_models(self.config.resource)

    def collect_all_loaded_data(self):
        state_ary_list, policy_ary_list, value_ary_list = [], [], []
        for s_ary, p_ary, v_ary in self.loaded_data.values():
            state_ary_list.append(s_ary)
            policy_ary_list.append(p_ary)
            value_ary_list.append(v_ary)
        state_ary = np.concatenate(state_ary_list)
        policy_ary = np.concatenate(policy_ary_list)
        value_ary = np.concatenate(value_ary_list)
        return state_ary, policy_ary, value_ary

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])

    def load_model(self):
        from chess_zero.agent.model_chess import ChessModel
        model = ChessModel(self.config)
        logger.debug(f"loading newest model")
        if not load_newest_model_weight(self.config.resource, model):
            raise RuntimeError(f"newest model cannot be loaded!")
        return model

    def load_play_data(self):
        filenames = get_game_data_filenames(self.config.resource)
        updated = False
        for filename in filenames:
            if filename in self.loaded_filenames:
                continue
            self.load_data_from_file(filename)
            updated = True

        for filename in (self.loaded_filenames - set(filenames)):
            self.unload_data_of_file(filename)
            updated = True

        if updated:
            logger.debug("updating training dataset")
            try:
                self.dataset = self.collect_all_loaded_data()
            except Exception as e:
                logger.warning(str(e))

    def load_data_from_file(self, filename):
        try:
            logger.debug(f"loading data from {filename}")
            data = read_game_data_from_file(filename)
            self.loaded_data[filename] = self.convert_to_training_data(data)
            self.loaded_filenames.add(filename)
        except Exception as e:
            logger.warning(str(e))

    def unload_data_of_file(self, filename):
        logger.debug(f"removing data about {filename} from training set")
        self.loaded_filenames.remove(filename)
        if filename in self.loaded_data:
            del self.loaded_data[filename]

    def convert_to_training_data(self, data):  # this method is no longer static.
        """

        :param data: format is SelfPlayWorker.buffer
        :return:
        """
        state_list = []
        policy_list = []
        value_list = []
        for state, policy, value in data:
            env = ChessEnv().update(state)

            state = env.gather_features(self.config.model.t_history)
            state_list.append(state)
            policy_list.append(policy)
            value_list.append(value)

        return np.array(state_list), np.array(policy_list), np.array(value_list)
