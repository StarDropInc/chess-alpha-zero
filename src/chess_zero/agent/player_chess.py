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

CounterKey = namedtuple("CounterKey", "board next_player")
QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit")

logger = getLogger(__name__)


class ChessPlayer:
    def __init__(self, config: Config, model, play_config=None):

        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.api = ChessModelAPI(self.config, self.model)

        self.move_lookup = {k:v for k,v in zip((chess.Move.from_uci(move) for move in self.config.labels), range(len(self.config.labels)))}
        self.labels_n = config.n_labels
        self.var_n = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_w = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_q = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_u = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_p = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.expanded = set()
        self.now_expanding = set()
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

        self.thinking_history = {}  # for fun

    def action(self, board):
        env = ChessEnv().update(board)
        key = self.counter_key(env)

        for tl in range(self.play_config.thinking_loop):
            if tl > 0 and self.play_config.logging_thinking:
                logger.debug(f"continue thinking: policy move=({action % 8}, {action // 8}), value move=({action_by_value % 8}, {action_by_value // 8})")
            if env.num_pieces() <= 5:  # syzygy takes over at this point, to generate training data of optimal quality.
                policy = self.syzygy_policy(board)  # note: in the essentially impossible situation under which num_pieces <= 5 before the change_tau_turn move, this will violate the temperature...
            else:
                self.search_moves(board)
                policy = self.calc_policy(board)
            action = int(np.random.choice(range(self.labels_n), p=policy))
            action_by_value = int(np.argmax(self.var_q[key] + (self.var_n[key] > 0)*100))  # what is the point of action_by_value?
            if action == action_by_value or env.turn < self.play_config.change_tau_turn:
                break

        # this is for play_gui, not necessary when training.
        self.thinking_history[env.observation] = HistoryItem(action, policy, list(self.var_q[key]), list(self.var_n[key]))

        if self.play_config.resign_threshold is not None and \
           self.play_config.min_resign_turn < env.turn and \
           env.absolute_eval(self.retrieve_eval(env.observation)) <= self.play_config.resign_threshold:
            return None  # means resign
        else:
            self.moves.append([env.observation, list(policy)])
            return self.config.labels[action]

    def ask_thought_about(self, board) -> HistoryItem:
        return self.thinking_history.get(board)

    def retrieve_eval(self, observation):
        last_history = self.ask_thought_about(observation)
        last_evaluation = last_history.values[last_history.action]
        return last_evaluation

    @profile
    def search_moves(self, board):
        start = time.time()
        loop = self.loop
        self.running_simulation_num = 0

        coroutine_list = []
        for it in range(self.play_config.simulation_num_per_move):
            cor = self.start_search_my_move(board)
            coroutine_list.append(cor)

        coroutine_list.append(self.prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))
        # logger.debug(f"Search time per move: {time.time()-start}")
        # uncomment to see profile result per move
        # raise

    async def start_search_my_move(self, board):
        self.running_simulation_num += 1
        with await self.sem:  # reduce parallel search number
            env = ChessEnv().update(board)
            leaf_v = await self.search_my_move(env, is_root_node=True)
            self.running_simulation_num -= 1
            return leaf_v

    async def search_my_move(self, env: ChessEnv, is_root_node=False):
        """

        Q, V is value for this Player (always white).
        P is value for the player of next_player (white or black)
        :param env:
        :param is_root_node:
        :return:
        """
        if env.done:
            if env.winner == Winner.WHITE:
                return 1
            elif env.winner == Winner.BLACK:
                return -1
            else:
                return 0

        key = self.counter_key(env)

        while key in self.now_expanding:
            await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)

        # is leaf?
        if key not in self.expanded:  # reach leaf node
            leaf_v = await self.expand_and_evaluate(env)
            return leaf_v if env.board.turn == chess.WHITE else -leaf_v

        action_t = self.select_action_q_and_u(env, is_root_node)

        env.step(self.config.labels[action_t])

        virtual_loss = self.config.play.virtual_loss
        self.var_n[key][action_t] += virtual_loss
        self.var_w[key][action_t] -= virtual_loss

        leaf_v = await self.search_my_move(env)  # next move

        # on returning search path
        # update: N, W, Q, U
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
        key = self.counter_key(env)
        self.now_expanding.add(key)

        white_ary, black_ary = env.white_and_black_plane()
        state = [white_ary, black_ary] if env.board.turn == chess.WHITE else [black_ary, white_ary]
        future = await self.predict(np.reshape(np.array(state), (12, 8, 8)))  # type: Future

        await future
        leaf_p, leaf_v = future.result()

        self.var_p[key] = leaf_p  # P is value for next_player (white or black)

        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

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
            #logger.debug(f"predicting {len(item_list)} items")
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

    def calc_policy(self, board):
        """calc π(a|s0)
        :return:
        """
        pc = self.play_config
        env = ChessEnv().update(board)
        key = self.counter_key(env)
        if env.turn < pc.change_tau_turn:
            return self.var_n[key] / (np.sum(self.var_n[key])+1e-8)  # tau = 1
        else:
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(self.labels_n)
            ret[action] = 1
            return ret

    def syzygy_policy(self, board):
        ret = np.zeros(self.labels_n)
        env = ChessEnv().update(board)
        with chess.syzygy.open_tablebases(self.config.resource.syzygy_dir) as tablebases:
            violent_wins = {}
            quiets_and_draws = {}
            violent_losses = {}
            for move in env.board.legal_moves:  # note: this scheme minimaxes distance to _zero_. distance to mate is not available through chess.syzygy
                env.board.push(move)
                dtz = float(tablebases.probe_dtz(env.board))
                value = 1/dtz if dtz != 0.0 else 0.0
                if env.board.halfmove_clock == 0 and value < 0:
                    violent_wins[move] = value
                elif env.board.halfmove_clock != 0 or value == 0:
                    quiets_and_draws[move] = value
                elif env.board.halfmove_clock == 0 and value > 0:
                    violent_losses[move] = value
                env.board.pop()
        if violent_wins:
            move = min(violent_wins, key=violent_wins.get)
        elif quiets_and_draws:
            move = min(quiets_and_draws, key=quiets_and_draws.get)
        elif violent_losses:
            move = min(violent_losses, key=violent_losses.get)
        action = self.move_lookup[move]
        ret[action] = 1
        return ret

    @staticmethod
    def counter_key(env: ChessEnv):
        return CounterKey(env.replace_tags(), env.board.turn)

    def select_action_q_and_u(self, env, is_root_node):
        key = self.counter_key(env)

        """Bottlenecks are these two lines"""
        legal_moves = [self.move_lookup[move] for move in env.board.legal_moves]
        legal_labels = np.zeros(len(self.config.labels))
        # logger.debug(legal_moves)
        legal_labels[legal_moves] = 1


        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[key]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
        p_ = self.var_p[key]

        if is_root_node:  # Is it correct?? -> (1-e)p + e*Dir(0.03)
            p_ = (1 - self.play_config.noise_eps) * p_ + \
                 self.play_config.noise_eps * np.random.dirichlet([self.play_config.dirichlet_alpha] * self.labels_n)

        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[key])
        if env.board.turn == chess.WHITE:
            v_ = (self.var_q[key] + u_ + 1000) * legal_labels
        else:
            # When enemy's selecting action, flip Q-Value.
            v_ = (-self.var_q[key] + u_ + 1000) * legal_labels

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t
