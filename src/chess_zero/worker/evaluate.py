import os
from datetime import datetime
from logging import getLogger
from random import random
from time import sleep
import chess
import chess.pgn
from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_old_model_dirs
from chess_zero.lib.model_helper import load_newest_model_weight
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed


logger = getLogger(__name__)


def start(config: Config):
    # tf_util.set_session_config(config.play.vram_frac)
    return EvaluateWorker(config).start()


class EvaluateWorker:
    def __init__(self, config: Config):
        """

        :param config:
        """
        self.config = config
        self.play_config = self.config.eval.play_config  # don't need other fields in self.eval...?
        self.current_model = ChessModel(self.config)
        self.m = Manager()
        self.current_pipes = self.m.list([self.current_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.play.max_processes)])

    def start(self):

        while True:
            load_newest_model_weight(self.config.resource, self.current_model)
            age = 0
            old_model, model_dir = self.load_old_model(age)  # how many models ago should we load?
            logger.debug(f"starting to evaluate newest model against model {model_dir}")
            newest_is_great = self.evaluate_model(old_model)
            if newest_is_great:
                logger.debug(f"the newest model defeated the {age}th archived model ({model_dir})")
            else:
                logger.debug(f"the newest model lost to the {age}th archived model ({model_dir})")

    def evaluate_model(self, old_model):
        old_pipes = self.m.list([old_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])
        with ProcessPoolExecutor(max_workers=self.play_config.max_processes) as executor:
            futures = [executor.submit(evaluate_buffer, self.config, self.current_pipes, old_pipes) for _ in range(self.config.eval.game_num)]
            results = []
            game_idx = 0
            for future in as_completed(futures):
                game_idx += 1
                current_win, env, current_is_white = future.result()  # why .get() as opposed to .result()?
                results.append(current_win)
                w = results.count(True)
                d = results.count(None)
                l = results.count(False)
                logger.debug(f"game {game_idx}: current won = {current_win} as {'White' if current_is_white else 'Black'}, W/D/L = {w}/{d}/{l}, {env.fen}")

                game = chess.pgn.Game.from_board(env.board)  # PGN dump
                game.headers['White'] = f"AI {self.current_model.digest[:10]}..." if current_is_white else f"AI {old_model.digest[:10]}..."
                game.headers['Black'] = f"AI {old_model.digest[:10]}..." if current_is_white else f"AI {self.current_model.digest[:10]}..."
                game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                logger.debug("\n" + str(game))

            return w / (w + l) >= self.config.eval.replace_rate

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


def evaluate_buffer(config, current, old) -> (float, ChessEnv, bool):
    current_pipes = current.pop()
    old_pipes = old.pop()

    random_endgame = config.eval.play_config.random_endgame
    if random_endgame == -1:
        env = ChessEnv(config).reset()
    else:
        env = ChessEnv(config).randomize(random_endgame)

    current_is_white = random() < 0.5

    current_player = ChessPlayer(config, pipes=current_pipes, play_config=config.eval.play_config)
    old_player = ChessPlayer(config, pipes=old_pipes, play_config=config.eval.play_config)

    while not env.done:
        ai = current_player if current_is_white == (env.board.turn == chess.WHITE) else old_player
        action = ai.action(env)
        env.step(action)

    current_win = None
    if env.winner != Winner.DRAW:
        current_win = current_is_white == (env.winner == Winner.WHITE)

    current.append(current_pipes)
    old.append(old_pipes)
    return current_win, env, current_is_white
