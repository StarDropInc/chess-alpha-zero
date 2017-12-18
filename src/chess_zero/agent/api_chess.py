from chess_zero.config import Config


class ChessModelAPI:
    def __init__(self, config: Config, model):
        self.config = config
        self.model = model

    def predict(self, x):
        assert x.ndim in (3, 4)
        input_stack_height = self.config.model.input_stack_height
        assert x.shape == (input_stack_height, 8, 8) or x.shape[1:] == (input_stack_height, 8, 8)  # should I get rid of these assertions...? they will change.
        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, input_stack_height, 8, 8)

        with self.model.graph.as_default():
            policy, value = self.model.model.predict_on_batch(x)

        if orig_x.ndim == 3:
            return policy[0], value[0]
        else:
            return policy, value
