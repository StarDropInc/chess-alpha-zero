class PlayDataConfig:
    def __init__(self):
        self.nb_game_in_file = 100  # 100
        self.max_file_num = 100  # 10


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 200  # 10
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 10  # 3
        self.noise_eps = .5  # .25
        self.dirichlet_alpha = 0.3
        self.tau_decay_rate = 0.99
        self.automatic_draw_turn = 100
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 16
        self.prediction_worker_sleep_sec = 0.00001
        self.wait_for_expanding_sleep_sec = 0.000001
        self.resign_threshold = None
        self.min_resign_turn = 10
        self.random_endgame = 5  # -1 for regular play, n > 2 for randomly generated endgames with n pieces.
        self.syzygy_access = False


class EvaluateConfig:
    def __init__(self):
        self.game_num = 100  # 10
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 100
        self.play_config.noise_eps = 0
        self.play_config.syzygy_access = False


class PlayWithHumanConfig:
    def __init__(self):
        self.play_config = PlayConfig()
        self.play_config.thinking_loop = 5
        self.play_config.logging_thinking = True
        self.play_config.syzygy_access = False


class TrainerConfig:
    def __init__(self):
        self.batch_size = 32  # 2048
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0
        self.save_model_steps = 2000
        self.load_data_steps = 1000
        self.min_data_size_to_learn = 10000


class ModelConfig:
    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_filter_size = 3
        self.res_layer_num = 19
        self.l2_reg = 1e-4
        self.value_fc_size = 265
        self.t_history = 8  # TEMP for small...
        self.input_stack_height = 7 + self.t_history*14
