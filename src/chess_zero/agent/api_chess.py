from chess_zero.config import Config
from threading import Thread
import numpy as np
import multiprocessing as mp
from multiprocessing import connection
import time

class ChessModelAPI:
    def __init__(self, model):
        self.model = model
        self.pipes = []

    def start(self):
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipe(self):
        me, you = mp.Pipe()
        self.pipes.append(me)
        return you

    def predict_batch_worker(self):
        while True:
            ready = mp.connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)
            if not data:
                continue
            data = np.asarray(data, dtype=np.float32)
            with self.model.graph.as_default():
                policy_ary, value_ary = self.model.model.predict_on_batch(data)
            for pipe, p, v in zip(result_pipes, policy_ary, value_ary):
                pipe.send((p, float(v)))
