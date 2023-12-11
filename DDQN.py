from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, SALTO_CORRER, SALTO_CORRER_AMBOS

import numpy as np
from collections import deque

from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import math
from itertools import count
import time
import collections

import matplotlib.pyplot as plt
import time
import os

import cv2
import numpy as np
import collections

from torch.utils.tensorboard import SummaryWriter


# Procesamiento del Ambiente
class MaxAndSkipEnv(gym.Wrapper):
	def __init__(self, env=None, skip=4):
		"""Return only every `skip`-th frame"""
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
		"""Clear past frame buffer and init to first obs"""
		self._obs_buffer.clear()
		obs = self.env.reset()
		self._obs_buffer.append(obs)
		return obs

class ProcessFrame84(gym.ObservationWrapper):
	"""
	Downsamples image to 84x84
	Greyscales image

	Returns numpy array
	"""
	def __init__(self, env=None):
		super(ProcessFrame84, self).__init__(env)
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

	def observation(self, obs):
		return ProcessFrame84.process(obs)

	@staticmethod
	def process(frame):
		if frame.size == 240 * 256 * 3:
			img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
		else:
			assert False, "Unknown resolution."

		img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
		resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
		x_t = resized_screen[18:102, :]
		x_t = np.reshape(x_t, [84, 84, 1])

		return x_t.astype(np.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
	def __init__(self, env):
		super(ImageToPyTorch, self).__init__(env)
		old_shape = self.observation_space.shape
		self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
												dtype=np.float32)

	def observation(self, observation):
		return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
	"""Normalize pixel values in frame --> 0 to 1"""
	def observation(self, obs):
		return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
	def __init__(self, env, n_steps, dtype=np.float32):
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

def wrap_env(env):
	env = MaxAndSkipEnv(env)
	env = ProcessFrame84(env)
	env = ImageToPyTorch(env)
	env = BufferWrapper(env, 4)
	env = ScaledFloatFrame(env)
	env = JoypadSpace(env, SALTO_CORRER)
	return env


# Replay Memory
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'new_state', 'done'])
class ReplayMemory:
	def __init__(self, max_size):
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)

	def push(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in idxs])
		return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
	
	def __len__(self):
		return len(self.buffer)
	

# DQN
class DDQN(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(DDQN, self).__init__()

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

class DDQNAgent:
	def __init__(self, env, render = True):
		self.env = env
		self.replay_buffer = ReplayMemory(max_size=BUFFER_SIZE)
		self.reset()
		self.render = render

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.policy = DDQN(env.observation_space.shape, env.action_space.n).to(self.device)
		self.target = DDQN(env.observation_space.shape, env.action_space.n).to(self.device)
		self.target.load_state_dict(self.policy.state_dict())
		self.target.eval()	# para que no actualize los pesos el optimizador ?

		# self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr = 1e-4 )
		# self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
		self.optimizer = optim.AdamW(self.policy.parameters(), lr=LEARNING_RATE, amsgrad=True)
		self.loss_fn = nn.SmoothL1Loss()

		# global
		self.envs = []
		for i in range(4):
			env = gym_super_mario_bros.make("SuperMarioBros-1-" + str(i+1) + "-v0")
			env = JoypadSpace(env, SALTO_CORRER) #SALTO_CORRER_AMBOS
			env = wrap_env(env)
			env.reset()
			self.envs.append(env)
	
	def reset(self):
		self.total_reward = 0.0
		self.state = self.env.reset()
		
	def play(self, eps):
		# random sample action
		if np.random.random() < eps:
			action = self.env.action_space.sample()

		# choose action with highest Q value
		else:
			state_tensor = torch.FloatTensor(self.state.astype(np.float32)).unsqueeze(0).to(self.device)
			qvals_tensor = self.policy(state_tensor)
			action = int(np.argmax(qvals_tensor.cpu().detach().numpy()))

		
		next_state, reward, done, _ = self.env.step(action)

		experience = Experience(self.state, action, reward, next_state, done)
		self.replay_buffer.push(experience)
		self.state = next_state
		self.total_reward += reward
		

		if self.render:
			self.env.render()

		if done:
			res_reward = self.total_reward
			self.reset()
			return res_reward
		else:
			return None
		
	def playGlobal(self, eps, ambiente):
		# epsilon-greedy
		if np.random.random() < eps:
			action = ambiente.action_space.sample()

		else:
			state_tensor = torch.FloatTensor(self.state.astype(np.float32)).unsqueeze(0).to(self.device)
			qvals_tensor = self.policy(state_tensor)
			action = int(np.argmax(qvals_tensor.cpu().detach().numpy()))

		
		next_state, reward, done, _ = ambiente.step(action)

		experience = Experience(self.state, action, reward, next_state, done)
		self.replay_buffer.push(experience)
		self.state = next_state
		self.total_reward += reward
		
		if self.render:
			ambiente.render()

		if done:
			res_reward = self.total_reward
			self.total_reward = 0.0

			return res_reward
		
		else:
			return None

	def optimizarModelo(self):
		if len(self.replay_buffer) < REPLAY_START:
			return

		# batch de experiencia y optimizacion de los datos
		states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

		states_tensor 		= torch.tensor(states.astype(np.float32), dtype=torch.float32).to(self.device)
		actions_tensor 		= torch.tensor(actions, dtype=torch.long).to(self.device)
		rewards_tensor 		= torch.tensor(rewards, dtype=torch.float32).to(self.device)
		next_states_tensor 	= torch.tensor(next_states.astype(np.float32), dtype=torch.float32).to(self.device)
		dones_tensor 		= torch.BoolTensor(dones).float().to(self.device)


		# DDQN
		# Compute Q-values for the current states using the online Q-network
		current_q_values = self.policy(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

		# Compute the target Q-values using the target Q-network
		with torch.no_grad():
			next_actions = self.policy(next_states_tensor).argmax(1)
			next_q_values = self.target(next_states_tensor).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
			target_q_values = rewards_tensor + GAMMA * next_q_values * (1 - dones_tensor)


		# Compute the loss using the Huber loss function
		loss = self.loss_fn(current_q_values, target_q_values)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.update_target()
	
	def update_target(self):
		for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
			target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)	

	def load_params(self, file_name):
		print(self.policy.load_state_dict(torch.load(file_name)))

		self.target.load_state_dict(self.policy.state_dict())
		self.target.eval()



def evaluacion(episode, agent, writer, last_Greddy, best_Greddy, carpeta):
	if episode%10 != 0:
		return last_Greddy, best_Greddy

	recompensasGreedy = []
	for l in MAPAS:
		env = gym_super_mario_bros.make("SuperMarioBros-1-" + str(l) + "-v0")
		env = wrap_env(env)
		agent.state = env.reset()
			
		while True:
			episode_reward = agent.playGlobal(0, env)

			if episode_reward is not None:
				recompensasGreedy.append(episode_reward)
				break

		env.close()
	
	promedio = np.mean(recompensasGreedy)
	writer.add_scalar("greedy", promedio, int(episode/10))

	if promedio > best_Greddy:
		best_Greddy = promedio

		if os.path.exists(last_Greddy):
			os.remove(last_Greddy)

		last_Greddy = carpeta + "DQN_G_" + str(promedio)
		torch.save(agent.policy.state_dict(), last_Greddy)
		

	return last_Greddy, best_Greddy

# Bucle entrenamiento
def entrenamiento(agent: DDQNAgent):
	episode_rewards = []
	eps = EPS_START
	total_steps = 0
	
	carpeta = str(math.floor(time.time())) + "/"
	writer = SummaryWriter(carpeta)

	best_mean_reward = 0
	last_DQN = ""

	best_Greddy = 0
	last_Greddy = ""
	
	for episode in range(EPISODES):
		last_Greddy, best_Greddy = evaluacion(episode, agent, writer, last_Greddy, best_Greddy, carpeta)

		env = gym_super_mario_bros.make("SuperMarioBros-1-" + str(MAPAS[episode % len(MAPAS)]) + "-v0")
		env = wrap_env(env)
		agent.state = env.reset()

		# entrenamiento real
		while True:
			total_steps += 1
			
			eps = max(eps*EPS_DECAY, EPS_END)
			episode_reward = agent.playGlobal(eps, env)
			agent.optimizarModelo()

			# si episode_reward = None entonces no ha terminado
			if episode_reward is not None:
				episode_rewards.append(episode_reward)
				mean_reward = np.mean(episode_rewards[-10:])

				if mean_reward > best_mean_reward:
					best_mean_reward = mean_reward

					if os.path.exists(last_DQN):
						os.remove(last_DQN)

					last_DQN = carpeta + "DQN_" + str(mean_reward)
					torch.save(agent.policy.state_dict(), last_DQN)
					print("Best mean reward updated, model saved")
					
				print("Episodio: %d | mean reward %.3f | steps: %d | epsilon %.3f" % (episode, mean_reward, total_steps,  eps))
				break
			
		env.close()

		writer.add_scalar("episode_reward", episode_reward, episode)
		writer.add_scalar("mean_reward", mean_reward, episode)
		writer.add_scalar("epsilon", eps, episode)


	writer.close()
	return episode_rewards




# ----------------------------------- Ambiente ------------------------------------
BATCH_SIZE          = 32				# numero de experiencias del buffer a usar
REPLAY_START        = BATCH_SIZE	    # steps minimos para empezar	
BUFFER_SIZE  		= 100000			# steps maximos que guarda 

TAU 				= 0.01				# actualizacion pasiva de la target network

EPISODES			= 10000   
GAMMA				= 0.90
EPS_START			= 1
EPS_END				= 0.001
EPS_DECAY 			= 0.9999
LEARNING_RATE		= 0.00025 		# 1e-4
MAPAS = [4]



def main():
	env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
	env = wrap_env(env)
	env.reset()

	# hiperparametros
	agent = DDQNAgent(env, render=False)
	#agent.load_params("DQN_1523.14_1")
	entrenamiento(agent)


if __name__ == "__main__":
   main()