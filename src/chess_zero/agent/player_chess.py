from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
from logging import getLogger
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

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


class ChessPlayer:
    def __init__(self, config: Config, model, play_config=None):

        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.api = ChessModelAPI(self.config, self.model)

        self.labels = self.config.labels
        self.n_labels = config.n_labels
        self.var_n = defaultdict(lambda: np.zeros((self.n_labels,)))
        self.var_w = defaultdict(lambda: np.zeros((self.n_labels,)))
        self.var_q = defaultdict(lambda: np.zeros((self.n_labels,)))
        self.var_p = defaultdict(lambda: np.zeros((self.n_labels,)))
        self.var_moves = {}  # dict storing the legal moves for each position.
        self.expanded = set()
        self.now_expanding = set()
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)
        self.tablebases = chess.syzygy.open_tablebases(self.config.resource.syzygy_dir)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

        self.thinking_history = {}  # for fun

    def action(self, env):
        key = env.transposition_key()

        for tl in range(self.play_config.thinking_loop):
            if tl > 0 and self.play_config.logging_thinking:
                logger.debug(f"continue thinking: policy move=({action % 8}, {action // 8}), value move=({action_by_value % 8}, {action_by_value // 8})")
            if self.play_config.syzygy_access and env.board.num_pieces() <= 5:  # syzygy takes over at this point, to generate training data of optimal quality.
                legal_moves = env.board.legal_moves  # a temporary hack. replicating the contents of _hash_moves (there are issues with calling an async from a sync).
                # logger.debug(legal_moves)
                legal_labels = np.zeros(self.n_labels)
                legal_labels[[self.labels[move] for move in legal_moves]] = 1
                self.var_moves[key] = legal_moves, legal_labels
                policy = self.syzygy_policy(env)  # note: returns an "all or nothing" policy, regardless of tau, etc.
            else:
                self.search_moves(env)  # this should leave env invariant!
                policy = self.calc_policy(env)
            action = int(np.random.choice(range(self.n_labels), p=policy))
            action_by_value = int(np.argmax(self.var_q[key] + (self.var_n[key] > 0)*100))  # what is the point of action_by_value?
            if action == action_by_value:  # or env.fullmove_number < self.play_config.change_tau_turn:
                break

        # this is for play_gui, not necessary when training.
        self.thinking_history[env.fen] = HistoryItem(action, policy, list(self.var_q[key]), list(self.var_n[key]))

        if self.play_config.resign_threshold is not None and \
           self.play_config.min_resign_turn < env.fullmove_number and \
           env.absolute_eval(self.retrieve_eval(env.fen)) <= self.play_config.resign_threshold:
            return chess.Move.null()  # means resign
        else:
            self.moves.append([env.fen, list(policy)])
            legal_moves, _ = self.var_moves[key]
            move = next(move for move in legal_moves if self.labels[move] == action)
            return move

    def ask_thought_about(self, fen) -> HistoryItem:
        return self.thinking_history.get(fen)

    def retrieve_eval(self, fen):
        last_history = self.ask_thought_about(fen)
        last_evaluation = last_history.values[last_history.action]
        return last_evaluation

    @profile
    def search_moves(self, env):
        start = time.time()
        loop = self.loop
        self.running_simulation_num = 0

        coroutine_list = []
        for it in range(self.play_config.simulation_num_per_move):
            cor = self.start_search_my_move(env)
            coroutine_list.append(cor)

        coroutine_list.append(self.prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))
        # logger.debug(f"Search time per move: {time.time()-start}")
        # uncomment to see profile result per move
        # raise

    async def start_search_my_move(self, env):
        self.running_simulation_num += 1
        with await self.sem:  # reduce parallel search number
            my_env = env.copy()  # use this option to preserve history... but it's slow...!
            # my_env = ChessEnv(self.config).update(env.fen)
            leaf_v = await self.search_my_move(my_env, is_root_node=True)
            self.running_simulation_num -= 1
            return leaf_v  # it would appear that this return has no purpose.

    async def search_my_move(self, env: ChessEnv, is_root_node=False):
        """

        Q, V is value for this Player (always white).
        P is value for the player of next_player (white or black)
        :param env:
        :param is_root_node:
        :return:
        """
        if env.done:  # should an MCTS worker even have access to mate info...?
            if env.winner == Winner.WHITE:
                return 1
            elif env.winner == Winner.BLACK:
                return -1
            else:
                return 0

        key = env.transposition_key()

        if self.play_config.syzygy_access and env.board.num_pieces() <= 5:  # syzygy bases can guide the internal MCTS search as well, not just the high-level moves.
            while key in self.now_expanding:
                await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)
            if key not in self.expanded:
                leaf_v = await self.syzygy_and_evaluate(env)
                return leaf_v if env.board.turn == chess.WHITE else -leaf_v
            action_t, move_t = self.select_action_syzygy(env)
        else:
            while key in self.now_expanding:
                await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)
            if key not in self.expanded:  # reach leaf node
                leaf_v = await self.expand_and_evaluate(env)
                return leaf_v if env.board.turn == chess.WHITE else -leaf_v
            action_t, move_t = self.select_action_q_and_u(env, is_root_node)

        env.step(move_t)

        virtual_loss = self.config.play.virtual_loss
        self.var_n[key][action_t] += virtual_loss
        self.var_w[key][action_t] -= virtual_loss

        leaf_v = await self.search_my_move(env)  # next move

        # on returning search path
        # update: N, W, Q
        n = self.var_n[key][action_t] = self.var_n[key][action_t] - virtual_loss + 1
        w = self.var_w[key][action_t] = self.var_w[key][action_t] + virtual_loss + leaf_v
        self.var_q[key][action_t] = w / n

        return leaf_v

    @profile
    async def expand_and_evaluate(self, env):
        """expand new leaf

        update var_p, return leaf_v

        :param ChessEnv env:
        :return: leaf_v
        """
        key = env.transposition_key()
        self.now_expanding.add(key)

        state = env.board.gather_features(self.config.model.t_history)
        future = await self.predict(state)  # type: Future

        await future
        leaf_p, leaf_v = future.result()

        legal_moves, legal_labels = await self._hash_moves(env)
        leaf_p = leaf_p * legal_labels  # mask policy vector and renormalize.
        leaf_p = leaf_p / sum(leaf_p) if sum(leaf_p) > 0 else leaf_p
        self.var_p[key] = leaf_p

        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    async def syzygy_and_evaluate(self, env):
        key = env.transposition_key()
        self.now_expanding.add(key)

        wdl = self.tablebases.probe_wdl(env.board)
        if wdl == 2:
            leaf_v = 1
        elif wdl == -2:
            leaf_v = -1
        else:
            leaf_v = 0

        await self._hash_moves(env)  # is this a good place for this to happen...?
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    async def _hash_moves(self, env):
        key = env.transposition_key()
        legal_moves = env.board.legal_moves
        # logger.debug(legal_moves)
        legal_labels = np.zeros(self.n_labels)
        legal_labels[[self.labels[move] for move in legal_moves]] = 1
        self.var_moves[key] = legal_moves, legal_labels
        return legal_moves, legal_labels

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.

        speed up about 45sec -> 15sec for example.
        :return:
        """
        q = self.prediction_queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.config.play.prediction_worker_sleep_sec)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            # logger.debug(f"predicting {len(item_list)} items")
            data = np.array([x.state for x in item_list])
            policy_ary, value_ary = self.api.predict(data)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def predict(self, x):
        future = self.loop.create_future()
        item = QueueItem(x, future)
        await self.prediction_queue.put(item)
        return future

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
        key = env.transposition_key()
        tau = np.power(pc.tau_decay_rate, env.fullmove_number)
        if tau <= 0.1:  # avoid numerical errors...?
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(self.n_labels)
            ret[action] = 1
            return ret
        else:
            n_exp = np.power(self.var_n[key], 1/tau)
            return n_exp / np.sum(n_exp)  # should never be dividing by 0

    def syzygy_policy(self, env):
        action, _ = self.select_action_syzygy(env)
        ret = np.zeros(self.n_labels)
        ret[action] = 1
        return ret

    def select_action_q_and_u(self, env, is_root_node):  # now returns a chess.Move, as opposed to an index
        key = env.transposition_key()
        legal_moves, legal_labels = self.var_moves[key]  # node has already been expanded and evaluated before this routine is called.

        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[key]))  # SQRT of sum(N(s, b); for all b)
        xx_ += 1  # avoid u_=0 if N is all 0  # WAS max(xx_, 1)... avoid a discontinuity...!
        p_ = self.var_p[key]

        if is_root_node:  # Is it correct? -> (1-e)p + e*Dir(0.03)
            p_ = (1 - self.play_config.noise_eps) * p_ + self.play_config.noise_eps * np.random.dirichlet([self.play_config.dirichlet_alpha] * self.n_labels)

        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[key])

        v_ = ((1 if env.board.turn == chess.WHITE else -1) * self.var_q[key] + u_ + 1000) * legal_labels
        # under extreme bad luck, the vector p_, and thus u_, could become entirely negative after dirichlet noise. need the argmaxing _legal_ index, even if negative...

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        move_t = next(move for move in legal_moves if self.labels[move] == action_t) # this iterator must yield exactly one item! the restriction of _labels_ to _legal_moves_ is injective.
        return action_t, move_t

    def select_action_syzygy(self, env):
        key = env.transposition_key()
        legal_moves, _ = self.var_moves[key]  # node has already been (syzygied) and evaluated before this routine is called.

        violent_wins = {}
        quiets_and_draws = {}
        violent_losses = {}
        for move in legal_moves:  # note: node probably _hasn't_ been expanded. this move generation could probably be stored, but...
            is_zeroing = env.board.is_zeroing(move)
            env.board.push(move)  # note: minimizes distance to _zero_. distance to mate is not available through the syzygy bases. but gaviota are much larger...
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
        return action_t, move_t
