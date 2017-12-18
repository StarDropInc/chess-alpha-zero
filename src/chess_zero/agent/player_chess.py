from collections import defaultdict, namedtuple
from logging import getLogger
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Thread, Lock

from profilehooks import profile

import time
import numpy as np
import chess

from chess_zero.agent.api_chess import ChessModelAPI
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner

QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit")

logger = getLogger(__name__)


# these are from AGZ nature paper
class VisitStats:
    def __init__(self):
        self.a = defaultdict(ActionStats)  # (key, value) of type (Move, ActionStats)?
        self.sum_n = 0
        self.selected_yet = False


class ActionStats:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0


class ChessPlayer:
    def __init__(self, config: Config, model=None, play_config=None):

        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.api = ChessModelAPI(self.config, self.model)

        self.labels = self.config.labels
        self.n_labels = config.n_labels
        self.tablebases = chess.syzygy.open_tablebases(self.config.resource.tablebase_dir)
        self.prediction_queue_lock = Lock()
        self.is_thinking = False

        self.moves = []

        self.reset()

    def reset(self):
        self.tree = defaultdict(VisitStats)
        self.node_lock = defaultdict(Lock)
        self.prediction_queue = []

    def action(self, env):
        self.reset()

        key = env.transposition_key()

        self.is_thinking = True
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

        try:  # what exceptions do you think will be thrown here?
            for tl in range(self.play_config.thinking_loop):
                if self.play_config.tablebase_access and env.board.num_pieces() <= 5:  # tablebase takes over
                    policy = self.tablebase_policy(env)  # note: returns an "all or nothing" policy, regardless of tau, etc.
                else:
                    self.search_moves(env)  # this should leave env invariant!
                    policy = self.calc_policy(env)
                action = int(np.random.choice(range(self.n_labels), p=policy))
        finally:
            self.is_thinking = False

        if self.play_config.resign_threshold is not None and \
           self.play_config.min_resign_turn < env.fullmove_number and \
           np.max([a_s.q for a_s in self.tree[key].a.values()]) <= self.play_config.resign_threshold:  # technically, resigning should be determined by leaf_v.
            return chess.Move.null()  # means resign
        else:
            self.moves.append([env.fen, list(policy)])
            move = next(move for move in self.tree[key].a.keys() if self.labels[move] == action)
            return move

    def sl_action(self, env, move):
        ret = np.zeros(self.n_labels)
        action = self.labels[move]
        ret[action] = 1

        self.moves.append([env.fen, list(ret)])
        return move

    def search_moves(self, env):
        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.parallel_search_num) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move, env=env.copy(), is_root_node=True))
        [f.result() for f in futures]

    def search_my_move(self, env: ChessEnv, is_root_node=False):
        """

        Q, V is value for this Player (always white).
        P is value for the player of next_player (white or black)
        :param env:
        :param is_root_node:
        :return: leaf value
        """
        if env.done:  # should an MCTS worker even have access to mate info...?
            if env.winner == Winner.DRAW:
                return 0
            else:
                return -1  # a tricky optimization: this conditional will _only_ execute if the side to move has just lost.

        key = env.transposition_key()

        my_lock = self.node_lock[key]

        with my_lock:
            if key not in self.tree:
                leaf_v = self.expand_and_evaluate(env)
                return leaf_v  # I'm returning everything from the POV of side to move
            # keep the same lock open?
            move_t, action_t = self.select_action_q_and_u(env, is_root_node)

        env.step(move_t)

        virtual_loss = self.play_config.virtual_loss
        with my_lock:
            my_visitstats = self.tree[key]
            my_actionstats = my_visitstats.a[move_t]

            my_visitstats.sum_n += virtual_loss
            my_actionstats.n += virtual_loss
            my_actionstats.w += -virtual_loss

        leaf_v = -self.search_my_move(env)  # next move

        # on returning search path, update: N, W, Q
        with my_lock:
            my_visitstats.sum_n += -virtual_loss + 1
            my_actionstats.n += -virtual_loss + 1
            my_actionstats.w += virtual_loss + leaf_v
            my_actionstats.q = my_actionstats.w / my_actionstats.n

        return leaf_v

    def expand_and_evaluate(self, env) -> float:
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

        self.tree[env.transposition_key()].temp_p = leaf_p

        return float(leaf_v)

    def tablebase_and_evaluate(self, env):
        wdl = self.tablebases.probe_wdl(env.board)
        if wdl == 2:
            leaf_v = 1
        elif wdl == -2:
            leaf_v = -1
        else:
            leaf_v = 0

        return float(leaf_v)

    def predict_batch_worker(self):
        while self.is_thinking:
            if self.prediction_queue:
                with self.prediction_queue_lock:
                    item_list = self.prediction_queue  # doesn't this just copy the reference?
                    self.prediction_queue = []

                # logger.debug(f"predicting {len(item_list)} items")
                data = np.array([x.state for x in item_list])
                policy_ary, value_ary = self.api.predict(data)
                for item, p, v in zip(item_list, policy_ary, value_ary):
                    item.future.set_result((p, v))
            else:
                time.sleep(self.play_config.prediction_worker_sleep_sec)

    def predict(self, state):
        future = Future()
        item = QueueItem(state, future)
        with self.prediction_queue_lock:  # lists are atomic anyway though
            self.prediction_queue.append(item)
        return future.result()

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

        my_visitstats = self.tree[env.transposition_key()]
        var_n = np.zeros(self.n_labels)
        var_n[[self.labels[move] for move in my_visitstats.a.keys()]] = [a_s.n for a_s in my_visitstats.a.values()]  # too 'pythonic'? a.keys() and a.values() are guaranteed to be in the same order.

        if env.fullmove_number < pc.change_tau_turn:
            return var_n / my_visitstats.sum_n  # should never be dividing by 0
        else:
            action = np.argmax(var_n)  # tau = 0
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

        my_visitstats = self.tree[env.transposition_key()]
        if not my_visitstats.selected_yet:
            my_visitstats.selected_yet = True
            tot_p = 0
            for move in env.board.legal_moves:
                move_p = my_visitstats.temp_p[self.labels[move]]
                my_visitstats.a[move].p = move_p  # defaultdict is key here.
                tot_p += move_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p

        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # SQRT of sum(N(s, b); for all b)

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dirichlet_alpha = self.play_config.dirichlet_alpha

        v_ = {move:(a_s.q + c_puct * (a_s.p if not is_root_node else (1 - e) * a_s.p + e * np.random.dirichlet([dirichlet_alpha])) * xx_ / (1 + a_s.n)) for move, a_s in my_visitstats.a.items()}  # too much on one line...?
        move_t = max(v_, key=v_.get)
        action_t = self.labels[move_t]

        return move_t, action_t

    def select_action_tablebase(self, env):
        key = env.transposition_key()

        violent_wins = {}
        quiets_and_draws = {}
        violent_losses = {}
        for move in env.board.legal_moves:  # note: not worrying about hashing this.
            is_zeroing = env.board.is_zeroing(move)
            env.board.push(move)  # note: minimizes distance to _zero_. distance to mate is not available through the tablebase bases. but gaviota are much larger...
            dtz = self.tablebases.probe_dtz(env.board)  # casting to float isn't necessary; is coerced below upon comparison to 0.0
            value = 1/dtz if dtz != 0.0 else 0.0  # a trick: fast mated < slow mated < draw < slow mate < fast mate
            if is_zeroing and value < 0:
                violent_wins[move] = value
            elif not is_zeroing or value == 0:
                quiets_and_draws[move] = value
            elif is_zeroing and value > 0:
                violent_losses[move] = value
            env.board.pop()
        if violent_wins:
            move_t = min(violent_wins, key=violent_wins.get)
        elif quiets_and_draws:
            move_t = min(quiets_and_draws, key=quiets_and_draws.get)
        elif violent_losses:
            move_t = min(violent_losses, key=violent_losses.get)
        action_t = self.labels[move_t]
        return move_t, action_t

