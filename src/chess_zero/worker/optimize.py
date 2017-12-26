from logging import getLogger
from time import sleep
from time import time

import numpy as np
import os
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import chess

from chess_zero.config import Config
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from chess_zero.lib.model_helper import load_newest_model_weight, save_as_newest_model, clear_old_models
from chess_zero.env.chess_env import MyBoard
from chess_zero.agent.model_chess import ChessModel
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(config.trainer.vram_frac)
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
        last_load_data_step = last_save_step = total_steps = 0
        self.load_play_data()

        while True:
            if self.dataset_size < tc.min_data_size_to_learn:
                logger.info(f"dataset_size={self.dataset_size} is less than {tc.min_data_size_to_learn}")
                sleep(60)
                self.load_play_data()
                continue
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
        state_deque, policy_deque, value_deque = self.dataset
        state_ary, policy_ary, value_ary = np.asarray(state_deque), np.asarray(policy_deque), np.asarray(value_deque)
        tensorboard_cb = TensorBoard(log_dir=self.config.resource.log_dir, batch_size=tc.batch_size, histogram_freq=1)
        self.model.model.fit(state_ary, [policy_ary, value_ary], batch_size=tc.batch_size, epochs=epochs, shuffle=True, validation_split=0.05, callbacks=[tensorboard_cb])
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        self.optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def replace_current_model(self):
        save_as_newest_model(self.config.resource, self.model)
        clear_old_models(self.config.resource)

    def load_model(self):
        model = ChessModel(self.config)
        if self.config.opts.new or not load_newest_model_weight(self.config.resource, model):
            model.build()  # optimize will now _also_ build a new model from scratch if none exists.
            save_as_newest_model(self.config.resource, model)
        return model

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])

    def load_play_data(self):
        new_filenames = set(get_game_data_filenames(self.config.resource)[-self.config.trainer.max_num_files_in_memory:])

        for filename in self.loaded_filenames - new_filenames:
            logger.debug(f"removing data {filename} from training set")
            self.loaded_filenames.remove(filename)
            del self.loaded_data[filename]

        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            futures = {executor.submit(load_data_from_file, filename, self.config.model.t_history):filename for filename in new_filenames - self.loaded_filenames}
            for future in as_completed(futures):
                filename = futures[future]
                logger.debug(f"loading data from {filename}")
                self.loaded_filenames.add(filename)
                self.loaded_data[filename] = future.result()

        self.dataset = self.collect_all_loaded_data()

    def collect_all_loaded_data(self):
        if not self.loaded_data:
            return
        state_ary_list, policy_ary_list, value_ary_list = [], [], []
        for s_ary, p_ary, v_ary in self.loaded_data.values():
            state_ary_list.extend(s_ary)
            policy_ary_list.extend(p_ary)
            value_ary_list.extend(v_ary)
        state_ary = np.stack(state_ary_list)
        policy_ary = np.stack(policy_ary_list)
        value_ary = np.expand_dims(np.stack(value_ary_list), axis=1)
        return state_ary, policy_ary, value_ary


def load_data_from_file(filename, t_history):
    # necessary to catch an exception here...? if the play data file isn't completely written yet, then some error will be thrown about a "missing delimiter", etc.
    data = read_game_data_from_file(filename)

    state_list = []
    policy_list = []
    value_list = []

    board = MyBoard(None)
    board.fullmove_number = 1000  # an arbitrary large value.

    for state, policy, value in data:
        board.push_fen(state)
        state = board.gather_features(t_history)
        if board.turn == chess.BLACK:
            policy = Config.flip_policy(policy)

        state_list.append(state)
        policy_list.append(policy)
        value_list.append(value)

    return state_list, policy_list, value_list
