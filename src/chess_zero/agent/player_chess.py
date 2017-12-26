from collections import defaultdict, namedtuple
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import chess

from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
# import chess.gaviota
# import chess.syzygy

QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit")

logger = getLogger(__name__)


# these are from AGZ nature paper
class VisitStats:
    def __init__(self):
        self.a = defaultdict(ActionStats)  # (key, value) of type (Move, ActionStats)?
        self.sum_n = 0


class ActionStats:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0


class ChessPlayer:
    def __init__(self, config: Config, pipes=None, dummy=False, play_config=None):

        self.config = config
        self.play_config = play_config or self.config.play

        self.labels = self.config.labels
        self.n_labels = config.n_labels
        # self.tablebases = chess.gaviota.open_tablebases(self.config.resource.tablebase_dir)
        # self.tablebases = chess.syzygy.open_tablebases(self.config.resource.tablebase_dir)
        self.moves = []
        if dummy:
            return

        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)
        self.reset()

    def reset(self):
        self.tree = defaultdict(VisitStats)

    def action(self, env):
        self.reset()

        if self.play_config.tablebase_access and env.board.num_pieces() <= 5:  # tablebase takes over
            root_value = self.tablebase_and_evaluate(env)
            policy = self.tablebase_policy(env)  # note: returns an "all or nothing" policy, regardless of tau, etc.
        else:
            root_value = self.search_moves(env)  # this should leave env invariant!
            policy = self.calc_policy(env)
        action = int(np.random.choice(range(self.n_labels), p=policy))

        if self.play_config.resign_threshold is not None and \
           self.play_config.min_resign_turn < env.fullmove_number and \
           root_value <= self.play_config.resign_threshold:  # technically, resigning should be determined by leaf_v?
            return chess.Move.null()  # means resign
        else:
            self.moves.append([env.fen, list(policy)])
            move = next(move for move in self.tree[env.transposition_key()].a.keys() if self.labels[move] == action)
            return move

    def sl_action(self, env, move):
        ret = np.zeros(self.n_labels)
        action = self.labels[move]
        ret[action] = 1

        self.moves.append([env.fen, list(ret)])
        return move

    def search_moves(self, env) -> (float, float):
        num_sims = self.play_config.simulation_num_per_move
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            vals = executor.map(self.search_my_move, [env.copy() for _ in range(num_sims)], [True for _ in range(num_sims)])

        return np.max(vals)

    def search_my_move(self, env: ChessEnv, is_root_node):
        """

        Q, V is value for this Player (always white).
        P is value for the player of next_player (white or black)
        :param env:
        :param is_root_node:
        :return: leaf value
        """
        if env.done:
            if env.winner == Winner.DRAW:
                return 0
            else:
                return -1  # a tricky optimization: this conditional will _only_ execute if the side to move has just lost.

        key = env.transposition_key()

        with self.node_lock[key]:
            if key not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[key].p = leaf_p
                return leaf_v  # returning everything from the POV of side to move
            # keep the same lock open?
            move_t, action_t = self.select_action_q_and_u(env, is_root_node)

            virtual_loss = self.play_config.virtual_loss
            my_visit_stats = self.tree[key]
            my_action_stats = my_visit_stats.a[move_t]
            my_visit_stats.sum_n += virtual_loss
            my_action_stats.n += virtual_loss
            my_action_stats.w += -virtual_loss
            my_action_stats.q = my_action_stats.w / my_action_stats.n  # fixed a bug: must update q here...


        env.step(move_t)
        leaf_v = -self.search_my_move(env, False)  # next move

        # on returning search path, update: N, W, Q
        with self.node_lock[key]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_action_stats.n += -virtual_loss + 1
            my_action_stats.w += virtual_loss + leaf_v
            my_action_stats.q = my_action_stats.w / my_action_stats.n

        return leaf_v

    def expand_and_evaluate(self, env) -> (np.ndarray, float):
        """expand new leaf

        this is called with state locked
        insert P(a|s), return leaf_v

        :param ChessEnv env:
        :return: leaf_v
        """
        if self.play_config.tablebase_access and env.board.num_pieces() <= 5:
            return self.tablebase_and_evaluate(env)

        state = env.board.gather_features(self.config.model.t_history)
        leaf_p, leaf_v = self.predict(state)

        if env.board.turn == chess.BLACK:
            leaf_p = Config.flip_policy(leaf_p)

        return leaf_p, leaf_v

    def tablebase_and_evaluate(self, env):
        wdl = self.tablebases.probe_wdl(env.board)
        # under syzygy, wdl can be 2 or -2. (1 and -1 are cursed win and blessed loss, respectively.)
        if wdl >= 1:
            leaf_v = 1
        elif wdl <= 1:
            leaf_v = -1
        else:
            leaf_v = 0

        return float(leaf_v)

    def predict(self, state):
        pipe = self.pipe_pool.pop()
        pipe.send(state)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    def finish_game(self, z):
        """

        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]

    def calc_policy(self, env):
        """calc π(a|s0)
        :return:
        """
        pc = self.play_config

        my_visit_stats = self.tree[env.transposition_key()]
        policy = np.zeros(self.n_labels)
        policy[[self.labels[move] for move in my_visit_stats.a.keys()]] = [a_s.n for a_s in my_visit_stats.a.values()]  # too 'pythonic'? a.keys() and a.values() are guaranteed to be in the same order.

        if env.fullmove_number < pc.change_tau_turn:
            return policy / my_visit_stats.sum_n  # should never be dividing by 0
        else:
            action = np.argmax(policy)  # tau = 0
            ret = np.zeros(self.n_labels)
            ret[action] = 1
            return ret

    def tablebase_policy(self, env):
        _, action = self.select_action_tablebase(env)
        ret = np.zeros(self.n_labels)
        ret[action] = 1
        return ret

    def select_action_q_and_u(self, env, is_root_node):
        # this method is called with state locked
        if self.play_config.tablebase_access and env.board.num_pieces() <= 5:
            return self.select_action_tablebase(env)

        my_visit_stats = self.tree[env.transposition_key()]
        if my_visit_stats.p is not None:
            tot_p = 0
            for move in env.board.legal_moves:
                move_p = my_visit_stats.p[self.labels[move]]
                my_visit_stats.a[move].p = move_p  # defaultdict is key here.
                tot_p += move_p
            for a_s in my_visit_stats.a.values():
                a_s.p /= tot_p
            my_visit_stats.p = None

        xx_ = np.sqrt(my_visit_stats.sum_n + 1)  # SQRT of sum(N(s, b); for all b)

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dirichlet_alpha = self.play_config.dirichlet_alpha

        v_ = {move:(a_s.q + c_puct * (a_s.p if not is_root_node else (1 - e) * a_s.p + e * np.random.dirichlet([dirichlet_alpha])) * xx_ / (1 + a_s.n)) for move, a_s in my_visit_stats.a.items()}  # too much on one line...?
        move_t = max(v_, key=v_.get)
        action_t = self.labels[move_t]

        return move_t, action_t

    def select_action_tablebase(self, env):
        moves = {}
        for move in env.board.legal_moves:
            env.board.push(move)
            dtm = self.tablebases.probe_dtm(env.board)
            value = 1/dtm if dtm != 0.0 else 0.0
            moves[move] = value
            env.board.pop()
        move_t = min(moves, key=moves.get)
        action_t = self.labels[move_t]
        return move_t, action_t

    # Uncomment the below code if using the SYZYGY TABLEBASES. NOTE: syzygy only provides _distance to zero_, which in general does not coincide with optimal _distance to mate_.
    # def select_action_tablebase(self, env):
    #     violent_wins = {}
    #     quiets_and_draws = {}
    #     violent_losses = {}
    #     for move in env.board.legal_moves:  # note: not worrying about hashing this.
    #         is_zeroing = env.board.is_zeroing(move)
    #         env.board.push(move)  # note: minimizes distance to _zero_. distance to mate is not available through the tablebase bases. but gaviota are much larger...
    #         dtz = self.tablebases.probe_dtz(env.board)  # casting to float isn't necessary; is coerced below upon comparison to 0.0
    #         value = 1/dtz if dtz != 0.0 else 0.0  # a trick: fast mated < slow mated < draw < slow mate < fast mate
    #         if is_zeroing and value < 0:
    #             violent_wins[move] = value
    #         elif not is_zeroing or value == 0:
    #             quiets_and_draws[move] = value
    #         elif is_zeroing and value > 0:
    #             violent_losses[move] = value
    #         env.board.pop()
    #     if violent_wins:
    #         move_t = min(violent_wins, key=violent_wins.get)
    #     elif quiets_and_draws:
    #         move_t = min(quiets_and_draws, key=quiets_and_draws.get)
    #     elif violent_losses:
    #         move_t = min(violent_losses, key=violent_losses.get)
    #     action_t = self.labels[move_t]
    #     return move_t, action_t
