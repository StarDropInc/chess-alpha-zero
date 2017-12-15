from logging import getLogger

import chess
from chess_zero.config import Config
from chess_zero.gui.game_model import PlayWithHuman
from chess_zero.env.chess_env import ChessEnv
from random import random

logger = getLogger(__name__)


def start(config: Config):
    chess_model = PlayWithHuman(config)

    env = ChessEnv(config).reset()
    human_is_white = random() < 0.5
    chess_model.start_game(human_is_white)

    while not env.done:
        if (env.board.turn == chess.WHITE) == human_is_white:
            action = chess_model.move_by_human(env)
            print(f"You move to: {env.board.san(action)}")
        else:
            action = chess_model.move_by_ai(env)
            print(f"AI moves to: {env.board.san(action)}")
        env.step(action)
        print(env.board)
        print(f"Board FEN = {env.fen}")

    print(f"\nEnd of the game. Game result: {env.board.result()}")
