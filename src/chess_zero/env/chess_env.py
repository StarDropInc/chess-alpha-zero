import enum
import chess.pgn
import chess.syzygy
import numpy as np

from logging import getLogger
import random

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "WHITE BLACK DRAW")


class ChessEnv:
    def __init__(self):
        self.board = None
        self.done = False
        self.winner = None  # type: Winner
        self.resigned = False

    def reset(self):
        self.board = chess.Board()
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        self.board = chess.Board(board)
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def randomize(self, num):  # generates a random position with _num_ pieces on the board. used to generate training data (with tablebase)
        self.board = chess.Board.empty()
        self.done = False
        self.winner = None
        self.resigned = False

        squares = random.sample(range(0, 64), num)
        self.board.set_piece_at(squares[0], chess.Piece(6, chess.WHITE))
        self.board.set_piece_at(squares[1], chess.Piece(6, chess.BLACK))
        self.board.set_piece_at(squares[2], chess.Piece(random.randrange(1, 6), random.random() < 0.5))
        self.board.set_piece_at(squares[3], chess.Piece(random.randrange(1, 6), random.random() < 0.5))
        self.board.set_piece_at(squares[4], chess.Piece(random.randrange(1, 6), random.random() < 0.5))

        if not self.board.is_valid() or self.board.is_game_over():  # possible if the randomly generated position is a (stale)mate! note: could replace with setting self.done appropriately.
            return self.randomize(num)

        return self

    def step(self, action):
        """
        :param int|None action, None is resign
        :return:
        """
        if action is None:
            self._resigned()
            return

        self.board.push_uci(action)

        if self.board.is_game_over() or self.board.can_claim_draw():
            self._game_over()

    def _game_over(self):
        self.done = True
        result = self.board.result()
        if result == '1/2-1/2' or self.board.can_claim_draw():
            self.winner = Winner.DRAW
        else:
            self.winner = Winner.WHITE if result == '1-0' else Winner.BLACK


    def absolute_eval(self, relative_eval):
        return relative_eval if self.board.turn == chess.WHITE else -relative_eval

    def _resigned(self):
        self.winner = Winner.BLACK if self.board.turn == chess.WHITE else Winner.WHITE
        self.done = True
        self.resigned = True

    def num_pieces(self):
        board_state = self.replace_tags()
        return len([val for val in board_state.split(" ")[0] if val != "1"])

    def white_and_black_planes(self):
        board_state = self.replace_tags()

        one_hot = {}
        one_hot.update(dict.fromkeys(['K', 'k'], [1, 0, 0, 0, 0, 0]))
        one_hot.update(dict.fromkeys(['Q', 'q'], [0, 1, 0, 0, 0, 0]))
        one_hot.update(dict.fromkeys(['R', 'r'], [0, 0, 1, 0, 0, 0]))
        one_hot.update(dict.fromkeys(['B', 'b'], [0, 0, 0, 1, 0, 0]))
        one_hot.update(dict.fromkeys(['N', 'n'], [0, 0, 0, 0, 1, 0]))
        one_hot.update(dict.fromkeys(['P', 'p'], [0, 0, 0, 0, 0, 1]))

        board_white = [one_hot[val] if val.isupper() else [0, 0, 0, 0, 0, 0] for val in board_state.split(" ")[0]]
        board_white = np.transpose(np.reshape(board_white, (8, 8, 6)), (2, 0, 1))
        board_black = [one_hot[val] if val.islower() else [0, 0, 0, 0, 0, 0] for val in board_state.split(" ")[0]]
        board_black = np.transpose(np.reshape(board_black, (8, 8, 6)), (2, 0, 1))

        return board_white, board_black

    def replace_tags(self):
        board_san = self.board.fen()
        board_san = board_san.split(" ")[0]
        board_san = board_san.replace("2", "11")
        board_san = board_san.replace("3", "111")
        board_san = board_san.replace("4", "1111")
        board_san = board_san.replace("5", "11111")
        board_san = board_san.replace("6", "111111")
        board_san = board_san.replace("7", "1111111")
        board_san = board_san.replace("8", "11111111")

        return board_san.replace("/", "")

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    @property
    def observation(self):
        return self.board.fen()

    @property
    def turn(self):
        return self.board.fullmove_number
