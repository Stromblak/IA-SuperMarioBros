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
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

	def observation(self, obs):
		return ProcessFrame128.process(obs)

	@staticmethod
	def process(frame):
		if frame.size == 240 * 256 * 3:
			img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
		else:
			assert False, "Unknown resolution."

		img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
		resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
		x_t = np.reshape(resized_screen[18:102, :], [84, 84, 1])
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
	def __init__(self, env, buffer_size=100000, render = False):
		self.env = env
		self.render = render
		self.replay_buffer = ReplayMemory(max_size=buffer_size)
		self.reset()

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.policy = DDQN(env.observation_space.shape, env.action_space.n).to(self.device)
		self.target = DDQN(env.observation_space.shape, env.action_space.n).to(self.device)
		self.target.load_state_dict(self.policy.state_dict())

		self.target.eval()	# para que no actualize los pesos el optimizador ?

		# self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr = 1e-4 )
		# self.optimizer = optim.Adam(self.policy.parameters(), lr=0.learning_rate)
		self.optimizer = optim.AdamW(self.policy.parameters(), lr=learning_rate, amsgrad=True)

		self.loss_func = nn.SmoothL1Loss()
	
	def reset(self):
		self.total_reward = 0.0
		self.state = self.env.reset()
		
	def play(self, eps):
		if self.render: self.env.render()
	
		# random sample action
		if np.random.random() < eps:
			action = self.env.action_space.sample()

		# choose action with highest Q value
		else:
			state_tensor = torch.FloatTensor(self.state.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
			qvals_tensor = self.policy(state_tensor)
			action = int(np.argmax(qvals_tensor.cpu().detach().numpy()))

		
		next_state, reward, done, _ = self.env.step(action)

		experience = Experience(self.state, action, reward, next_state, done)
		self.replay_buffer.push(experience)
		self.state = next_state
		self.total_reward += reward
		
		if done:
			res_reward = self.total_reward
			self.reset()
			return res_reward
		else:
			return None

	def optimizarModelo(self):
		if len(self.replay_buffer) < replay_start:
			return

		# batch de experiencia y optimizacion de los datos
		states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

		states_tensor 		= torch.tensor(states.astype(np.float32) / 255.0, dtype=torch.float32).to(self.device)
		actions_tensor 		= torch.tensor(actions, dtype=torch.long).to(self.device)
		rewards_tensor 		= torch.tensor(rewards, dtype=torch.float32).to(self.device)
		next_states_tensor 	= torch.tensor(next_states.astype(np.float32) / 255.0, dtype=torch.float32).to(self.device)
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
		loss = F.smooth_l1_loss(current_q_values, target_q_values)
	

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.update_target()


	def load_params(self, file_name):
		print(self.policy.load_state_dict(torch.load(file_name)))
	
	def update_target(self):
		metodo = 0

		if metodo == 0:
			# Soft update of the target network's weights
			target_net_state_dict = self.target.state_dict()
			policy_net_state_dict = self.policy.state_dict()

			for key in policy_net_state_dict:
				target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

			self.target.load_state_dict(target_net_state_dict)

		elif metodo == 1:
			for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
				target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)	



# Entrenamiento

def entrenamiento(agent, writer, expected_reward = 2000, replay_start = 16384):
	episode_rewards = []
	eps = EPS_START
	# Total steps played 
	total_steps = 0
	best_mean_reward = 1000.0
	
	model_no = 0
	for episode in range(1, EPISODES + 1):
		for step in count():
			total_steps += 1
			
			eps = max(eps*EPS_DECAY, EPS_END)
			#eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)

			episode_reward = agent.play(eps)

			# si episode_reward = None entonces no ha terminado
			if episode_reward is not None:
				episode_rewards.append(episode_reward)
				mean_reward = np.mean(episode_rewards[-100:])
				writer.add_scalar("epsilon", eps, total_steps)
				writer.add_scalar("last_100_reward", mean_reward, total_steps)
				writer.add_scalar("reward", episode_reward, total_steps)

				if mean_reward > best_mean_reward and total_steps > replay_start:
					best_mean_reward = mean_reward
					# save the most recent 10 best models
					torch.save(agent.model.state_dict(), "Local_DQN_Mario_no" + str(model_no) + "_" + str(mean_reward))
					print("Best mean reward updated, model saved")
					model_no += 1
					model_no = model_no % 10
					

				print("Episodio: %d | mean reward %.3f | Steps: %d | epsilon %.2f" % (episode, mean_reward, total_steps,  eps))
				break
			
			agent.optimizarModelo()
	
		# Guardar el modelo
		if episode % 1000 == 0 and episode > 0:
			torch.save(agent.model.state_dict(), "Local_DQN_Mario_snapshot_" + str(episode))
			print("snapshot saved")	

		if best_mean_reward >= expected_reward:
			break


	torch.save(agent.model.state_dict(), "models/Local_DQN_Mario_snapshot_" + str(math.floor(time.time())))

	writer.close()
	return episode_rewards




# ----------------------------------- Ambiente ------------------------------------
BATCH_SIZE          = 32			# numero de experiencias del buffer a usar
replay_start        = BATCH_SIZE	# steps minimos para empezar  #10000		
BUFFER_SIZE  		= 30000			# steps maximos que guarda 

TAU 				= 0.005			# actualizacion pasiva de la target network

EPISODES			= 10000   
GAMMA				= 0.90
EPS_START			= 1.0
EPS_END				= 0.05
EPS_DECAY 			= 0.999999
recompensa_termino	= 3000.0
learning_rate		= 0.00025 		# 1e-4

def main():
	env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
	env = JoypadSpace(env, SALTO_CORRER) #SIMPLE_MOVEMENT
	env = wrap_env(env)
	env.reset()


	# hiperparametros


	writer = SummaryWriter('runs/' + str(math.floor(time.time())) + '-mario_1_1-' + str(EPS_DECAY) + '-' + str(BUFFER_SIZE))

	agent = DDQNAgent(env, BUFFER_SIZE, render=True)
	#agent.load_params("Local_DQN_Mario_snapshot_1")

	episode_rewards = entrenamiento(agent, 
								writer			= writer, 
								expected_reward	= recompensa_termino, 
								replay_start	= replay_start)




	def grafico(nombre, rewards):
		n = 100
		smoothed_rewards = []
		for i in range(len(rewards)):
			start = max(0, i - n)
			end = min(len(rewards), i + n + 1)
			smoothed_rewards.append(sum(rewards[start:end]) / (end - start))

		plt.plot(smoothed_rewards, label=nombre)
		plt.ylabel('Recompensa')
		plt.xlabel('Episodio')
		plt.legend()
		
		plt.savefig(nombre + ".png", format='png')

		file = open('rewards.txt','w')
		for r in rewards:
			file.write(str(r) + "\n")

		file.close()

	grafico("DQN", episode_rewards)
	