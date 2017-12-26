import os
from datetime import datetime
from logging import getLogger
from time import time, sleep
import chess
from concurrent.futures import ProcessPoolExecutor, as_completed
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.agent.model_chess import ChessModel
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from chess_zero.lib.model_helper import load_newest_model_weight, save_as_newest_model
from multiprocessing import Manager
from collections import deque
from threading import Thread

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(config.play.vram_frac)
    return SelfPlayWorker(config).start()

class SelfPlayWorker:
    def __init__(self, config: Config):
        """

        :param config:
        :param ChessEnv|None env:
        :param chess_zero.agent.model_chess.ChessModel|None model:
        """
        self.config = config
        self.current_model = self.load_model()
        self.m = Manager()
        self.current_pipes = self.m.list([self.current_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.play.max_processes)])

    def start(self):
        self.buffer = []
        load_newest_model_weight(self.config.resource, self.current_model)

        game_idx = 0
        while True:
            with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
                futures = [executor.submit(self_play_buffer, self.config, self.current_pipes) for _ in range(self.config.play_data.nb_game_in_file)]
                start_time = time()
                for future in as_completed(futures):
                    game_idx += 1
                    env, data = future.result()
                    logger.debug(f"game {game_idx} time={(time() - start_time):.3f}s, turn={int(env.fullmove_number)}. {env.winner}, resigned: {env.resigned}, {env.fen}")
                    start_time = time()
                    self.buffer += data
            self.flush_buffer()
            load_newest_model_weight(self.config.resource, self.current_model)
            self.remove_play_data()

    def load_model(self):
        model = ChessModel(self.config)
        if self.config.opts.new or not load_newest_model_weight(self.config.resource, model):
            model.build()
            save_as_newest_model(self.config.resource, model)
        return model

    def flush_buffer(self):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []

    def remove_play_data(self):
        filenames = get_game_data_filenames(self.config.resource)[:-self.config.play_data.max_file_num]
        for filename in filenames:
            os.remove(filename)


def self_play_buffer(config, current) -> (ChessEnv, list):
    pipes = current.pop()  # borrow

    random_endgame = config.eval.play_config.random_endgame
    if random_endgame == -1:
        env = ChessEnv(config).reset()
    else:
        env = ChessEnv(config).randomize(random_endgame)

    white = ChessPlayer(config, pipes=pipes)
    black = ChessPlayer(config, pipes=pipes)

    while not env.done:
        ai = white if env.board.turn == chess.WHITE else black
        action = ai.action(env)
        env.step(action)

    if env.winner == Winner.WHITE:
        white_win = 1
    elif env.winner == Winner.BLACK:
        white_win = -1
    else:
        white_win = 0

    white.finish_game(white_win)
    black.finish_game(-white_win)

    # game = chess.pgn.Game.from_board(env.board)  # optional PGN dump
    # game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    # game.headers['White'] = game.headers['Black'] = f"AI {self.model.digest[:10]}..."

    current.append(pipes)
    return env, merge_data(white, black)


def merge_data(white, black):
    data = [move for pair in zip(white.moves, black.moves) for move in pair]  # interleave the two lists
    if len(white.moves) > len(black.moves):
        data += [white.moves[-1]]  # tack on final move if white moved last

    return data
