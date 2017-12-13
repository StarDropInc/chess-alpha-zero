class EvaluateConfig:
    def __init__(self):
        self.game_num = 100  # 400
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 800
        self.play_config.c_puct = 1
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0
        self.play_config.syzygy_access = False


class PlayDataConfig:
    def __init__(self):
        self.nb_game_in_file = 100
        self.max_file_num = 200  # 5000


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 800
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 2
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3
        self.change_tau_turn = 20  # 10
        self.automatic_draw_turn = 40
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 16
        self.prediction_worker_sleep_sec = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001
        self.resign_threshold = None  # -1.0
        self.min_resign_turn = 10
        self.syzygy_access = True


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
        self.res_layer_num = 19  # was 7, why? should this be 19 or 39?
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.t_history = 8
        self.input_stack_height = 7 + self.t_history*14
