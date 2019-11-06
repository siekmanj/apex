----

Recurrent Reinforcement Learning

## Running experiments

### Basics
Any algorithm can be run from the rll.py entry point.

To run DDPG on Walker2d-v2,

```bash
python rll.py ddpg --env_name Walker2d-v2 --batch_size 64
```

### Logging details / Monitoring live training progress
Tensorboard logging is enabled by default for all algorithms. The logger expects that you supply an argument named ```logdir```, containing the root directory you want to store your logfiles in, and an argument named ```seed```, which is used to seed the pseudorandom number generators.

A basic command line script illustrating this is:

```bash
python rll.py ars --logdir logs/ars --seed 1337
```

The resulting directory tree would look something like this:
```
logs/
├── ars
│   └── <env_name> 
│           └── [New Experiment Logdir]
├── ddpg
└── rdpg
```

Using tensorboard makes it easy to compare experiments and resume training later on.

To see live training progress

Run ```$ tensorboard --logdir=logs``` then navigate to ```http://localhost:6006/``` in your browser

### To Do
- [ ] Recurrent TD3 and normal TD3

### Notes

Troubleshooting: X module not found? Make sure PYTHONPATH is configured. Make sure you run 
examples from root directory.

## Features:
* Parallelism with [Ray](https://github.com/ray-project/ray)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [RDPG](https://arxiv.org/abs/1512.04455)
* [ARS](https://arxiv.org/abs/1803.07055)

#### To be implemented long term:
* [SAC](https://arxiv.org/abs/1801.01290)
* [SVG](https://arxiv.org/abs/1510.09142)

## Acknowledgements

This repo was cloned from the Oregon State University DRL's Apex library: https://github.com/osudrl/apex (authored by my fellow researchers Yesh Godse and Pedro Morais), which was in turn inspired by @ikostrikov's implementations of RL algorithms. Thanks to @sfujim for the clean implementations of TD3 and DDPG in PyTorch. Thanks @modestyachts for the easy to understand ARS implementation.
