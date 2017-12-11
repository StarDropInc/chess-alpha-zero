import enum
import chess.pgn
import numpy as np

from logging import getLogger

import random
import collections
import chess.syzygy
from chess import Board
from chess import STARTING_FEN

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
        self.board = MyBoard()
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def update(self, fen):
        self.board = MyBoard(fen)
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def randomize(self, num):  # generates a random position with _num_ pieces on the board. used to generate training data (with tablebase)
        self.board = MyBoard(None)
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

    one_hot = {}  # is there a better way than to make this static and forward-declared...? the idea is to only construct this dictionary once.
    one_hot[None] = [0, 0, 0, 0, 0, 0]
    one_hot.update(dict.fromkeys(['P', 'p'], [1, 0, 0, 0, 0, 0]))
    one_hot.update(dict.fromkeys(['N', 'n'], [0, 1, 0, 0, 0, 0]))
    one_hot.update(dict.fromkeys(['B', 'b'], [0, 0, 1, 0, 0, 0]))
    one_hot.update(dict.fromkeys(['R', 'r'], [0, 0, 0, 1, 0, 0]))
    one_hot.update(dict.fromkeys(['Q', 'q'], [0, 0, 0, 0, 1, 0]))
    one_hot.update(dict.fromkeys(['K', 'k'], [0, 0, 0, 0, 0, 1]))

    @classmethod
    def _one_hot(cls, piece, side):  # is this static stuff a good idea...?
        key = piece.symbol() if piece != None and piece.color == side else None
        return cls.one_hot[key]

    def step(self, action):
        """
        :param int|None action, None is resign
        :return:
        """
        if action == chess.Move.null():
            self._resigned()
            return

        self.board.push(action)

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
        return sum([len(self.board.pieces(piece_type, color)) for piece_type in range(1,7) for color in [True, False]])

    def gather_features(self, t_history):  # t_history = T denotes the number of half-moves back into the game's history to go. In AlphaZero: T = 8
        stack = []
        stack.append(np.full((1, 8, 8), self.board.halfmove_clock))  # np.int64's will later be coerced into np.float64's.
        stack.append(np.full((1, 8, 8), self.board.has_queenside_castling_rights(False), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.board.has_kingside_castling_rights(False), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.board.has_queenside_castling_rights(True), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.board.has_kingside_castling_rights(True), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.board.fullmove_number))
        stack.append(np.full((1, 8, 8), self.board.turn, dtype=np.float64))
        self._recursive_append(stack, t_history - 1, self.board.turn)
        return np.concatenate(stack)

    def _recursive_append(self, stack, depth, side):
        if depth > 0:
            (move, fresh, fen) = chess.Move.null(), False, None
            if self.board.move_stack:  # there are still moves to pop.
                move = self.board.pop()
            elif self.board.is_valid():  # no more moves left, but still valid. we'll clear the board now
                fresh = True
                fen = self.board.fen()
                self.board.clear()
            self._recursive_append(stack, depth - 1, side)
            if fresh:
                self.board.set_fen(fen)
            else:
                self.board.push(move)

        repetitions = self.board.repetitions_count()
        stack.append(np.ones((1, 8, 8)) if repetitions >= 1 else np.zeros((1, 8, 8)))
        stack.append(np.ones((1, 8, 8)) if repetitions >= 2 else np.zeros((1, 8, 8)))

        board_enemy = [self._one_hot(self.board.piece_at(idx), not side) for idx in range(64)]
        board_enemy = np.transpose(np.reshape(board_enemy, (8, 8, 6)), (2, 0, 1))
        stack.append(np.flip(board_enemy, 1) if side else np.flip(board_enemy, 2))
        board_own = [self._one_hot(self.board.piece_at(idx), side) for idx in range(64)]
        board_own = np.transpose(np.reshape(board_own, (8, 8, 6)), (2, 0, 1))
        stack.append(np.flip(board_own, 1) if side else np.flip(board_own, 2))

    @property
    def fen(self):
        return self.board.fen()

    @property
    def fullmove_number(self):
        return self.board.fullmove_number

    @property
    def transposition_key(self):
        return self.board.transposition_key()


class MyBoard(Board):

    def __init__(self, fen=STARTING_FEN):
        Board.__init__(self, fen)

    def __str__(self):
        return self.unicode()

    def transposition_key(self):  # this feels slightly less egregious than accessing the private method from a completely foreign class.
        return self._transposition_key()

    def repetitions_count(self):  # essentially pilfered from python chess's _can_claim_threefold_repetition_ routine.
        transposition_key = self._transposition_key()
        transpositions = collections.Counter()
        transpositions.update((transposition_key, ))

        switchyard = collections.deque()
        while self.move_stack:
            move = self.pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            transpositions.update((self._transposition_key(), ))

        while switchyard:
            self.push(switchyard.pop())

        return transpositions[transposition_key]
