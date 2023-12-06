from gym_super_mario_bros import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, SALTO_CORRER

import random
import numpy as np
from collections import deque

from torchsummary import summary
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import gym
import math
from itertools import count
import time
import sys
import collections

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time


import cv2
import numpy as np
import collections

# Procesamiento del Ambiente

# Usar cada 4 frame

from DDQN import wrap_env, DQNAgent


# DQN
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SALTO_CORRER) #SIMPLE_MOVEMENT
env = wrap_env(env)
env.reset()


agent = DQNAgent(env, render=True)
agent.load_params("pesos")

for episode in range(10):
    print(episode)
    for step in count():
        episode_reward = agent.play(0)
