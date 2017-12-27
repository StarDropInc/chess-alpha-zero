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
from chess_zero.lib.data_helper import write_game_data_to_file, find_pgn_files
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Thread

logger = getLogger(__name__)

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")


def start(config: Config):
    return SupervisedLearningWorker(config, env=ChessEnv(config)).start()


class SupervisedLearningWorker:  # thanks to @Zeta36 and @Akababa for this class.
    def __init__(self, config: Config, env=None):
        """
        :param config:
        :param ChessEnv|None env:
        :param chess_zero.agent.model_chess.ChessModel|None model:
        """
        self.config = config

    def start(self):
        self.buffer = []
        start_time = time()

        with ProcessPoolExecutor(max_workers=8) as executor:
            games = self.get_games_from_all_files()
            game_idx = 0
            for future in as_completed([executor.submit(supervised_buffer, self.config, game) for game in games]):
                game_idx += 1
                env, data = future.result()
                self.buffer += data
                if game_idx % self.config.play_data.sl_nb_game_in_file == 0:
                    self.flush_buffer()
                end_time = time()
                logger.debug(f"game {game_idx} time={(end_time - start_time):.3f}s, turn={int(env.fullmove_number)}. {env.winner}, resigned: {env.resigned}, {env.fen}")
                start_time = end_time

    def get_games_from_all_files(self):
        files = find_pgn_files(self.config.resource.play_data_dir)
        games = []
        for filename in files:
            games.extend(self.get_games_from_file(filename))
        return games

    def get_games_from_file(self,filename):
        pgn = open(filename, errors='ignore')
        games = []
        for offset in chess.pgn.scan_offsets(pgn):
            pgn.seek(offset)
            games.append(chess.pgn.read_game(pgn))
        return games

    def flush_buffer(self):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"saving play data to {path}")
        thread = Thread(target = write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []


def supervised_buffer(config, game) -> (ChessEnv, list):
    env = ChessEnv(config).reset()
    white = ChessPlayer(config, dummy=True)
    black = ChessPlayer(config, dummy=True)
    result = game.headers["Result"]
    env.board = game.board()
    for move in game.main_line():
        ai = white if env.board.turn == chess.WHITE else black
        ai.sl_action(env, move)
        env.step(move)

    if not env.board.is_game_over() and result != '1/2-1/2':
        env.resigned = True
    if result == '1-0':
        env.winner = Winner.WHITE
        white_win = 1
    elif result == '0-1':
        env.winner = Winner.BLACK
        white_win = -1
    else:
        env.winner = Winner.DRAW
        white_win = 0

    white.finish_game(white_win)
    black.finish_game(-white_win)
    return env, merge_data(white, black)


def merge_data(white, black):
    data = [move for pair in zip(white.moves, black.moves) for move in pair]  # interleave the two lists
    if len(white.moves) > len(black.moves):
        data += [white.moves[-1]]  # tack on final move if white moved last

    return data
