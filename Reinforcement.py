import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9   # greedy policy
GAMMA = 0.9    # reward discount
TERGET_REPLACE_ITER = 100  # target updata frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = not env.action_space
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    None


