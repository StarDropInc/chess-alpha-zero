import enum
import chess.pgn
import numpy as np

from logging import getLogger

import random
import collections
import chess.syzygy
import copy
from chess import Board
from chess import STARTING_FEN
from chess_zero.config import Config

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "WHITE BLACK DRAW")


class ChessEnv:
    def __init__(self, config: Config):
        self.config = config
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
        for square in squares[2:]:
            self.board.set_piece_at(square, chess.Piece(random.randrange(1, 6), random.random() < 0.5))

        if not self.board.is_valid() or self.board.is_game_over():  # possible if the randomly generated position is a (stale)mate! note: could replace with setting self.done appropriately.
            return self.randomize(num)

        return self

    def step(self, move):
        """
        :param int|None action, None is resign
        :return:
        """
        if move == chess.Move.null():
            self._resign()
            return

        self.board.push(move)

        if self._is_game_over():
            self._conclude_game()

    def _is_game_over(self):
        return self.board.is_game_over() or self.board.can_claim_draw() or self.fullmove_number >= self.config.play.automatic_draw_turn

    def _conclude_game(self):
        self.done = True
        result = self.board.result()
        if result == '1/2-1/2' or self.board.can_claim_draw() or self.fullmove_number >= self.config.play.automatic_draw_turn:
            self.winner = Winner.DRAW
        else:
            self.winner = Winner.WHITE if result == '1-0' else Winner.BLACK

    def _resign(self):
        self.winner = Winner.BLACK if self.board.turn == chess.WHITE else Winner.WHITE
        self.done = True
        self.resigned = True

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env

    def transposition_key(self):  # used to be a @property, but that might be slower...?
        return self.board.transposition_key()

    @property
    def fen(self):
        return self.board.fen()

    @property
    def fullmove_number(self):
        return self.board.fullmove_number


class MyBoard(Board):

    def __init__(self, fen=STARTING_FEN):
        Board.__init__(self, fen)

    def __str__(self):
        return self.unicode()

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

    def transposition_key(self):  # this feels slightly less egregious than accessing the private method from a completely foreign class.
        return self._transposition_key()

    def num_pieces(self):
        return len(chess.SquareSet(self.occupied_co[chess.WHITE] | self.occupied_co[chess.BLACK]))

    def push_fen(self, fen):  # given a new FEN, either makes the corresponding move, or adopts the new FEN from scratch if there is none.
        new = chess.Board(fen)
        if self.turn != new.turn and self.fullmove_number <= new.fullmove_number:  # WARNING: this will break if a game lasts for just one move (i.e. is over when it starts)
            old_ss = chess.SquareSet(self.occupied_co[self.turn])
            new_ss = chess.SquareSet(new.occupied_co[self.turn])
            diff = list(new_ss.difference(old_ss))
            if len(diff) == 1:
                reverse = list(old_ss.difference(new_ss))
                move = chess.Move(reverse[0], diff[0])
                if self.piece_at(reverse[0]).piece_type != new.piece_at(diff[0]).piece_type:  # if a promotion occurred...
                    move.promotion = new.piece_at(diff[0]).piece_type
            elif len(diff) == 2:  # castling
                move = chess.Move(self.king(self.turn), new.king(self.turn))
            else:
                raise RuntimeError("problems with pushed fen.")
            self.push(move)
        else:
            self.set_fen(fen)

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

    def gather_features(self, t_history):  # t_history = T denotes the number of half-moves back into the game's history to go. In AlphaZero: T = 8
        stack = []
        stack.append(np.full((1, 8, 8), self.halfmove_clock))  # np.int64's will later be coerced into np.float64's.
        stack.append(np.full((1, 8, 8), self.has_queenside_castling_rights(False), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.has_kingside_castling_rights(False), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.has_queenside_castling_rights(True), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.has_kingside_castling_rights(True), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.fullmove_number))
        stack.append(np.full((1, 8, 8), self.turn, dtype=np.float64))
        self._recursive_append(stack, t_history - 1, self.turn)
        return np.concatenate(stack)

    def _recursive_append(self, stack, depth, side):
        if depth > 0:
            move, fen = chess.Move.null(), None
            if self.move_stack:  # there are still moves to pop.
                move = self.pop()
            elif self.is_valid():  # no more moves left, but still valid. we'll clear the board now
                fen = self.fen()
                self.clear()
            self._recursive_append(stack, depth - 1, side)
            if fen:
                self.set_fen(fen)
            else:
                self.push(move)

        repetitions = self.repetitions_count()
        stack.append(np.ones((1, 8, 8)) if repetitions >= 2 else np.zeros((1, 8, 8)))
        stack.append(np.ones((1, 8, 8)) if repetitions >= 3 else np.zeros((1, 8, 8)))

        board_enemy = [self._one_hot(self.piece_at(idx), not side) for idx in range(64)]
        board_enemy = np.transpose(np.reshape(board_enemy, (8, 8, 6)), (2, 0, 1))
        stack.append(np.flip(board_enemy, 1) if side else np.flip(board_enemy, 2))
        board_own = [self._one_hot(self.piece_at(idx), side) for idx in range(64)]
        board_own = np.transpose(np.reshape(board_own, (8, 8, 6)), (2, 0, 1))
        stack.append(np.flip(board_own, 1) if side else np.flip(board_own, 2))
