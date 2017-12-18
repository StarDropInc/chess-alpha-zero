import os
from datetime import datetime
from logging import getLogger
from time import time
import chess.pgn
import re
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file, find_pgn_files
from threading import Thread

import random

logger = getLogger(__name__)

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")


def start(config: Config):
    # tf_util.set_session_config(per_process_gpu_memory_fraction=0.01)
    return SupervisedLearningWorker(config, env=ChessEnv(config)).start()


class SupervisedLearningWorker:  # thanks to @Zeta36 and @Akababa for this class.
    def __init__(self, config: Config, env=None):
        """
        :param config:
        :param ChessEnv|None env:
        :param chess_zero.agent.model_chess.ChessModel|None model:
        """
        self.config = config
        self.env = env     # type: ChessEnv
        self.black = None  # type: ChessPlayer
        self.white = None  # type: ChessPlayer
        self.buffer = []
        self.idx = 1

    def start(self):
        start_time = time()

        for env in self.read_all_files():
            end_time = time()
            logger.debug(f"game {self.idx} time={(end_time - start_time):.3f}s, turn={int(env.fullmove_number)}. {env.winner}, resigned: {env.resigned}, {env.fen}")
            start_time = end_time
            self.idx += 1

        self.buffer = []

    def read_all_files(self):
        files = find_pgn_files(self.config.resource.play_data_dir)
        print (files)
        from itertools import chain
        return chain.from_iterable(self.read_file(filename) for filename in files)

    def read_file(self,filename):
        pgn = open(filename, errors='ignore')
        for offset, header in chess.pgn.scan_headers(pgn):
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            yield self.add_to_buffer(game)

    def add_to_buffer(self,game):
        self.env.reset()
        self.white = ChessPlayer(self.config)
        self.black = ChessPlayer(self.config)
        result = game.headers["Result"]
        self.env.board = game.board()
        for move in game.main_line():
            ai = self.white if self.env.board.turn == chess.WHITE else self.black
            ai.sl_action(self.env, move)
            self.env.step(move)

        self.env.done = True
        if not self.env.board.is_game_over() and result != '1/2-1/2':
            self.env.resigned = True
        if result == '1-0':
            self.env.winner = Winner.WHITE
        elif result == '0-1':
            self.env.winner = Winner.BLACK
        else:
            self.env.winner = Winner.DRAW

        self.finish_game()
        self.save_play_data()
        return self.env

    def save_play_data(self):
        data = [move for pair in zip(self.white.moves, self.black.moves) for move in pair]  # interleave the two lists
        if len(self.white.moves) > len(self.black.moves):
            data += [self.white.moves[-1]]  # tack on final move if white moved last
        self.buffer += data

        if self.idx % self.config.play_data.sl_nb_game_in_file == 0:
            self.flush_buffer()

    def flush_buffer(self):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        #print(self.buffer)
        thread = Thread(target = write_game_data_to_file, args=(path, (self.buffer)))
        thread.start()
        self.buffer = []

    def finish_game(self):
        if self.env.winner == Winner.WHITE:
            white_win = 1
        elif self.env.winner == Winner.BLACK:
            white_win = -1
        else:
            white_win = 0

        self.white.finish_game(white_win)
        self.black.finish_game(-white_win)
