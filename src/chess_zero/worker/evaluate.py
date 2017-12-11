import os
from logging import getLogger
from random import random
from time import sleep
import chess
from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_newest_model_dirs, get_old_model_dirs
from chess_zero.lib.model_helper import load_newest_model_weight

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.1)
    return EvaluateWorker(config).start()


class EvaluateWorker:
    def __init__(self, config: Config):
        """

        :param config:
        """
        self.config = config
        self.eval_config = self.config.eval
        self.newest_model = None

    def start(self):
        self.newest_model = self.load_newest_model()

        while True:
            self.refresh_newest_model()
            age = 0
            old_model, model_dir = self.load_old_model(age)  # how many models ago should we load?
            logger.debug(f"starting to evaluate newest model against model {model_dir}")
            newest_is_great = self.evaluate_model(old_model)
            if newest_is_great:
                logger.debug(f"the newest model defeated the {age}th archived model ({model_dir})")
            else:
                logger.debug(f"the newest model lost to the {age}th archived model ({model_dir})")

    def evaluate_model(self, old_model):
        results = []
        winning_rate = 0
        for game_idx in range(self.eval_config.game_num):
            # ng_win := if ng_model win -> 1, lose -> 0, draw -> None
            newest_win, newest_is_white = self.play_game(self.newest_model, old_model)
            if newest_win is not None:
                results.append(newest_win)
                winning_rate = sum(results) / len(results)
            logger.debug(f"game {game_idx}: newest won = {newest_win}, newest played white = {newest_is_white}, winning rate = {winning_rate*100:.1f}%")
            if results.count(0) >= self.eval_config.game_num * (1-self.eval_config.replace_rate):
                logger.debug(f"lose count has reached {results.count(0)}, so give up challenge")
                break
            if results.count(1) >= self.eval_config.game_num * self.eval_config.replace_rate:
                logger.debug(f"win count has reached {results.count(1)}, current model wins")
                break

        winning_rate = sum(results) / len(results) if len(results) != 0 else 0
        logger.debug(f"winning rate {winning_rate*100:.1f}%")
        return winning_rate >= self.eval_config.replace_rate

    def play_game(self, newest_model, old_model):
        env = ChessEnv().reset()
        # env = ChessEnv().randomize(5)

        newest_player = ChessPlayer(self.config, newest_model, play_config=self.eval_config.play_config)
        old_player = ChessPlayer(self.config, old_model, play_config=self.eval_config.play_config)
        newest_is_white = random() < 0.5

        while not env.done:
            ai = newest_player if newest_is_white == (env.board.turn == chess.WHITE) else old_player
            action = ai.action(env.fen)
            env.step(action)

        newest_win = None
        if env.winner != Winner.DRAW:
            newest_win = newest_is_white == (env.winner == Winner.WHITE)
        return newest_win, newest_is_white

    def load_newest_model(self):
        model = ChessModel(self.config)
        load_newest_model_weight(self.config.resource, model)
        return model

    def refresh_newest_model(self):
        load_newest_model_weight(self.config.resource, self.newest_model)

    def load_old_model(self, age):
        rc = self.config.resource
        while True:
            dirs = get_old_model_dirs(self.config.resource)
            if dirs:
                break
            logger.info(f"there is no old model to evaluate")
            sleep(60)
        model_dir = dirs[age]
        config_path = os.path.join(model_dir, rc.model_config_filename)
        weight_path = os.path.join(model_dir, rc.model_weight_filename)
        model = ChessModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir

    def remove_model(self, model_dir):
        rc = self.config.resource
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        os.remove(config_path)
        os.remove(weight_path)
        os.rmdir(model_dir)
