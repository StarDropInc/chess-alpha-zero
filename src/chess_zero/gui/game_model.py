from logging import getLogger

import tensorflow as tf
from chess_zero.agent.player_chess import HistoryItem
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.lib.model_helper import load_newest_model_weight
import chess

logger = getLogger(__name__)


class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.human_color = None
        self.observers = []
        self.model = self._load_model()
        self.ai = None  # type: ChessPlayer
        self.last_evaluation = None
        self.last_history = None  # type: HistoryItem

    def start_game(self, human_is_white):
        self.human_color = chess.WHITE if human_is_white else chess.BLACK
        self.ai = ChessPlayer(self.config, self.model, self.config.human.play_config)  # override self.config.play with optional third parameter

    def _load_model(self):
        from chess_zero.agent.model_chess import ChessModel
        model = ChessModel(self.config)
        if not load_newest_model_weight(self.config.resource, model):
            raise RuntimeError("newest model not found!")
        return model

    def move_by_ai(self, env):
        action = self.ai.action(env)

        return action

    def move_by_human(self, env):
        while True:
            san = input('\nEnter your move in SAN format ("e4", "Nf3", ... or "quit"): ')
            if san == "quit":
                raise SystemExit
            try:
                move = env.board.parse_san(san)
                if move != chess.Move.null():
                    return move
                else:
                    print("That is NOT a valid move :(.")  # how will parse_san ever return a null move...?
            except:
                print("That is NOT a valid move :(.")
