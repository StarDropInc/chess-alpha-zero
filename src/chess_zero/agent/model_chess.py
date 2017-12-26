import hashlib
import json
import os
from logging import getLogger
# noinspection PyPep8Naming

from tensorflow import get_default_graph
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from chess_zero.agent.api_chess import ChessModelAPI

from chess_zero.config import Config

logger = getLogger(__name__)


class ChessModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: Model
        self.graph = None
        self.digest = None
        self.api = None

    def get_pipes(self, num=1):
        if self.api is None:
            self.api = ChessModelAPI(self)
            self.api.start()
        return [self.api.get_pipe() for _ in range(num)]

    def build(self):
        mc = self.config.model
        in_x = x = Input((mc.input_stack_height, 8, 8))
        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same", data_format="channels_first", use_bias=False, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=l2(mc.l2_reg), input_shape=(mc.input_stack_height, 8, 8))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        for _ in range(mc.res_layer_num):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'
        policy_out = Dense(self.config.n_labels, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

        # for value output
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first", use_bias=False, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(mc.value_fc_size, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=l2(mc.l2_reg), activation="relu")(x)
        value_out = Dense(1, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="chess_model")
        self.graph = get_default_graph()

    def _build_residual_block(self, x):
        mc = self.config.model
        in_x = x
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same", data_format="channels_first", use_bias=False, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same", data_format="channels_first", use_bias=False, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path, weight_path):
        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"loading model from {config_path}")
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.graph = get_default_graph()
            # self.model._make_predict_function()
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"loaded model digest = {self.digest}")
            return True
        else:
            logger.debug(f"model files do not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path, weight_path):
        logger.debug(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")
