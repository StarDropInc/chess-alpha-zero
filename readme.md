About
=====

Chess reinforcement learning by [AlphaZero](https://arxiv.org/pdf/1712.01815.pdf) methods.

This project is based on the following resources:
1) DeepMind's Oct. 19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
2) DeepMind's recent arxiv paper [Mastering Chess and Shogi by Self-Play with a
General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf)
3) The <b>great</b> Reversi development of the DeepMind ideas that @mokemokechicken did in his repo: https://github.com/mokemokechicken/reversi-alpha-zero

Note: <b>This project is still under construction!!</b>

Environment
-----------

* Python 3.6.3
* tensorflow-gpu: 1.3.0
* Keras: 2.0.8

Modules
-------

### Reinforcement Learning

This AlphaZero implementation consists of two workers, `self` and `opt`.

* `self` plays the newest model against itself to generate self-play data for use in training.
* `opt` trains the existing model to create further new models, using the most recent self-play data.

### Evaluation

Evaluation options are provided by `eval` and `gui`.

* `eval` automatically tests the newest model by playing it against an older model (whose age can be specified).
* `gui` allows you to personally play against the newest model.

Data
-----

* `data/model/model_*`: newest model.
* `data/model/old_models/*`: archived old models.
* `data/play_data/play_*.json`: generated training data.
* `logs/main.log`: log file.
  
If you want to train a model from scratch, delete the above directories.

How to use
==========

Setup
-------
### install libraries
```bash
pip install -r requirements.txt
```

If you want use GPU,

```bash
pip install tensorflow-gpu
```

### set environment variables
Create `.env` file and write this.

```text:.env
KERAS_BACKEND=tensorflow
```


Basic Usage
------------

To train a model or further train an existing model, execute `Self-Play` and `Trainer`. 


Self-Play
--------

```bash
python src/chess_zero/run.py self
```

When executed, Self-Play will start using BestModel.
If the BestModel does not exist, new random model will be created and become BestModel.

### options
* `--new`: create new newest model from scratch
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)
* `--type small`: use small config for commodity hardware, (see `src/chess_zero/configs/small.py`)

Trainer
-------

```bash
python src/chess_zero/run.py opt
```

When executed, Training will start.
A base model will be loaded from latest saved next-generation model. If not existed, BestModel is used.
Trained model will be saved every 2000 steps(mini-batch) after epoch. 

### options
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)
* `--type small`: use small config for commodity hardware, (see `src/chess_zero/configs/small.py`)
* `--total-step`: specify an artificially nonzero starting point for total steps (mini-batches)

Evaluator
---------

```bash
python src/chess_zero/run.py eval
```

When executed, Evaluation will start.
It evaluates BestModel and the latest next-generation model by playing about 200 games.
If next-generation model wins, it becomes BestModel. 

### options
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)
* `--type small`: use small config for commodity hardware, (see `src/chess_zero/configs/small.py`)

Play Game
---------

```bash
python src/chess_zero/run.py gui
```

When executed, ordinary chess board will be displayed in unicode and you can play against the newest model.

### options
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)
* `--type small`: use small config for commodity hardware, (see `src/chess_zero/configs/small.py`)

Tips and Memos
====

GPU Memory
----------

Usually the lack of memory cause warnings, not error.
If error happens, try to change `per_process_gpu_memory_fraction` in `src/worker/{evaluate.py,optimize.py,self_play.py}`,

```python
tf_util.set_session_config(per_process_gpu_memory_fraction=0.2)
```

Less batch_size will reduce memory usage of `opt`.
Try to change `TrainerConfig#batch_size` in `NormalConfig`.

Syzygy Tablebases
-------
This implementation optionally uses the syzygy tablebases for endgame evaluation. The tablebase files should be placed into the directory chess-alpha-zero/syzygy. They can be generated from scratch through [the github repository](https://github.com/syzygy1/tb), or downloaded via the torrent "Syzygy 3-4-5 Individual Files" [here](http://oics.olympuschess.com/tracker/index.php).
