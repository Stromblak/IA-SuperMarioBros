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
class MaxAndSkipEnv(gym.Wrapper):
	def __init__(self, env=None, skip=4):
		super(MaxAndSkipEnv, self).__init__(env)
		# most recent raw observations (for max pooling across time steps)
		self._obs_buffer = collections.deque(maxlen=2)
		self._skip = skip

	def step(self, action):
		total_reward = 0.0
		done = None
		for _ in range(self._skip):
			obs, reward, done, info = self.env.step(action)
			self._obs_buffer.append(obs)
			total_reward += reward
			if done:
				break
		max_frame = np.max(np.stack(self._obs_buffer), axis=0)
		return max_frame, total_reward, done, info

	def reset(self):
		self._obs_buffer.clear()
		obs = self.env.reset()
		self._obs_buffer.append(obs)
		return obs

class ProcessFrame128(gym.ObservationWrapper):
	def __init__(self, env=None):
		super(ProcessFrame128, self).__init__(env)
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8)

	def observation(self, obs):
		return ProcessFrame128.process(obs)

	@staticmethod
	def process(frame):
		if frame.size == 240 * 256 * 3:
			img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
		else:
			assert False, "Unknown resolution."

		img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
		resized_screen = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
		x_t = np.reshape(resized_screen, [128, 128, 1])
		return x_t.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
	# def __init__(self, env, n_steps, dtype=np.float32):
	def __init__(self, env, n_steps, dtype=np.uint8):
		super(BufferWrapper, self).__init__(env)
		self.dtype = dtype
		old_space = env.observation_space
		self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
												old_space.high.repeat(n_steps, axis=0), dtype=dtype)

	def reset(self):
		self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
		return self.observation(self.env.reset())

	def observation(self, observation):
		self.buffer[:-1] = self.buffer[1:]
		self.buffer[-1] = observation
		return self.buffer

class ImageToPyTorch(gym.ObservationWrapper):
	def __init__(self, env):
		super(ImageToPyTorch, self).__init__(env)
		old_shape = self.observation_space.shape
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(old_shape[-1], 
								old_shape[0], old_shape[1]), dtype=np.uint8)
		# self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], 
		#                         old_shape[0], old_shape[1]), dtype=np.float32)        

	def observation(self, observation):
		return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
	def observation(self, obs):
		return np.array(obs).astype(np.uint8) 

def wrap_env(env):
	env = MaxAndSkipEnv(env)
	env = ProcessFrame128(env)
	env = ImageToPyTorch(env)
	env = BufferWrapper(env, 4)
	env = ScaledFloatFrame(env)
	return env



# DQN
class DQN(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(DQN, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU()
		)

		conv_out_size = self._get_conv_out(input_shape)
		self.fc = nn.Sequential(
			nn.Linear(conv_out_size, 512),
			nn.ReLU(),
			nn.Linear(512, n_actions)
		)

	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape))
		return int(np.prod(o.size()))

	def forward(self, x):
		conv_out = self.conv(x).view(x.size()[0], -1)
		return self.fc(conv_out)

class DQNAgent:
	def __init__(self, env):
		self.env = env
		self.reset()

		self.distancia = 0
		self.tiempo = 0
		self.atascado = False


		self.device = "cuda" if torch.cuda.is_available() else "cpu"


		self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
		self.target_model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
		self.target_model.load_state_dict(self.model.state_dict())
		self.target_model.eval()

		# self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = 1e-4 )
		# self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)
		self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, amsgrad=True)

		self.loss_func =  nn.SmoothL1Loss()
	
	def reset(self):
		self.total_reward = 0.0
		self.distancia = 0
		self.tiempo = 0

		self.state = self.env.reset()
		
	def play(self, eps):
		env.render()
	
		# random sample action
		action = 0
		if(np.random.random() < eps):
			action = self.env.action_space.sample()
		# choose action with highest Q value
		else:
			state_tensor = torch.FloatTensor(self.state.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
			# state_tensor = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
			qvals_tensor = self.model(state_tensor)
			action = int(np.argmax(qvals_tensor.cpu().detach().numpy()))

		
		next_state, reward, done, info = self.env.step(action)
		

		#reward = info["x_pos"] + info["y_pos"] + reward
		if info["x_pos"] > self.distancia:
			self.distancia = info["x_pos"]
			self.tiempo = info["time"]
		
		elif info["time"] == self.tiempo - 30:
			done = True

		self.state = next_state
		self.total_reward += reward
		
		if done:
			res_reward = self.total_reward
			self.reset()
			return res_reward
		else:
			return None
	
	def load_params(self, file_name):
		print(self.model.load_state_dict(torch.load(file_name)))
	
	def copy_to_target(self):
		print("Copying to target")
		self.target_model.load_state_dict(self.model.state_dict())

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SALTO_CORRER) #SIMPLE_MOVEMENT
env = wrap_env(env)
env.reset()


agent = DQNAgent(env)
agent.load_params("models\Local_DQN_Mario_snapshot_1699839468")

for episode in range(10):
    for step in count():
        episode_reward = agent.play(0.05)
