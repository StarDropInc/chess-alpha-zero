import os
from datetime import datetime
from logging import getLogger
from time import time
import chess
import tensorflow as tf
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from chess_zero.lib.model_helper import load_newest_model_weight, save_as_newest_model
from threading import Thread

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.4)
    return SelfPlayWorker(config, env=ChessEnv(config)).start()


class SelfPlayWorker:
    def __init__(self, config: Config, env=None, model=None):
        """

        :param config:
        :param ChessEnv|None env:
        :param chess_zero.agent.model_chess.ChessModel|None model:
        """
        self.config = config
        self.model = model
        self.env = env  # type: ChessEnv
        self.white = None  # type: ChessPlayer
        self.black = None  # type: ChessPlayer
        self.buffer = []
        self.idx = 1

    def start(self):
        if self.model is None:
            self.model = self.load_model()

        while True:
            start_time = time()
            env = self.start_game()
            end_time = time()
            logger.debug(f"game {self.idx} time={(end_time - start_time):.3f}s, turn={int(env.fullmove_number)}. {env.winner}, resigned: {env.resigned}, {env.fen}")
            if (self.idx % self.config.play_data.nb_game_in_file) == 0:
                load_newest_model_weight(self.config.resource, self.model)
            self.idx += 1

    def start_game(self):
        random_endgame = self.config.play.random_endgame
        if random_endgame == -1:
            self.env.reset()
        else:
            self.env.randomize(random_endgame)
        self.white = ChessPlayer(self.config, self.model)
        self.black = ChessPlayer(self.config, self.model)
        while not self.env.done:
            ai = self.white if self.env.board.turn == chess.WHITE else self.black
            move = ai.action(self.env)
            self.env.step(move)
        self.finish_game()
        game = chess.pgn.Game.from_board(self.env.board)
        game.headers['Event'] = f"Game {self.idx}"
        logger.debug("\n"+str(game))
        self.save_play_data()
        self.remove_play_data()
        return self.env

    def save_play_data(self):
        data = [move for pair in zip(self.white.moves, self.black.moves) for move in pair]  # interleave the two lists
        if len(self.white.moves) > len(self.black.moves):
            data += [self.white.moves[-1]]  # tack on final move if white moved last
        self.buffer += data

        if self.idx % self.config.play_data.nb_game_in_file == 0:
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

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        for i in range(len(files) - self.config.play_data.max_file_num):
            os.remove(files[i])

    def finish_game(self):
        if self.env.winner == Winner.WHITE:
            white_win = 1
        elif self.env.winner == Winner.BLACK:
            white_win = -1
        else:
            white_win = 0

        self.white.finish_game(white_win)
        self.black.finish_game(-white_win)

    def load_model(self):
        from chess_zero.agent.model_chess import ChessModel
        model = ChessModel(self.config)
        if self.config.opts.new or not load_newest_model_weight(self.config.resource, model):
            model.build()
            save_as_newest_model(self.config.resource, model)
        model.graph = tf.get_default_graph()
        return model
