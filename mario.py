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


from DDQN import wrap_env, DDQNAgent


# DQN
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SALTO_CORRER)
env = wrap_env(env)
env.reset()

agent = DDQNAgent(env)

agent.load_params("DDQN-DERECHA-1-MUNDO\Local_DQN_Mario_snapshot_9000")



for episode in range(1, 5):
    ambiente = (episode-1) % 4
    env = gym_super_mario_bros.make("SuperMarioBros-1-" + str(ambiente+1) + "-v0")
    env = JoypadSpace(env, SALTO_CORRER)
    env = wrap_env(env)
    agent.state = env.reset()
    
    for step in count():
        env.render()
        if agent.playGlobal(0, env):
            break
    
    env.close()

    