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



# Replay Memory
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'new_state'])
class ReplayMemory:
	def __init__(self, max_size):
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)

	def push(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, next_states = zip(*[self.buffer[i] for i in idxs])
		return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)
	
	def __len__(self):
		return len(self.buffer)
	


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
	def __init__(self, env, buffer_size=100000, render = False):
		self.env = env
		self.render = render
		self.replay_buffer = ReplayMemory(max_size=buffer_size)
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
		if self.render: env.render()
	
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


		experience = Experience(self.state, action, reward, next_state)

		self.replay_buffer.push(experience)
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


# Entrenamiento
def optimizarModelo(agent, tau, batch_size, gamma, update_interval):
	if len(agent.replay_buffer) < replay_start:
		return

	# batch de experiencia
	states, actions, rewards, next_states = agent.replay_buffer.sample(batch_size)

	# cosas misticas del tensor
	states_tensor = torch.tensor(states.astype(np.float32) / 255.0, dtype=torch.float32).to(agent.device)
	actions_tensor = torch.tensor(actions, dtype=torch.long).to(agent.device)
	rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(agent.device)
	next_states_tensor = torch.tensor(next_states.astype(np.float32) / 255.0, dtype=torch.float32).to(agent.device)

	# get current Q value from the training network given the current action
	curr_Q = agent.model(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

	# get expected Q value for all possible actions using the target net, and pick 
	# the highest Qval, e = reward + gamma * Q'(s_next)
	max_next_Q = agent.target_model(next_states_tensor).max(1)[0]

	expected_Q = rewards_tensor + gamma * max_next_Q.detach()

	loss = agent.loss_func(curr_Q, expected_Q)

	agent.optimizer.zero_grad()
	loss.backward()
	agent.optimizer.step()            


	# Soft update of the target network's weights
	target_net_state_dict = agent.target_model.state_dict()
	policy_net_state_dict = agent.model.state_dict()

	for key in policy_net_state_dict:
		target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)

	agent.target_model.load_state_dict(target_net_state_dict)


	# Actualizar la target network
	"""
	if total_steps % update_inverval == 0:
		agent.copy_to_target()
	"""

def entrenamiento(agent, 
				  max_episodes, 
				  writer, 
				  expected_reward = 150, 
				  replay_start = 16384, 
				  batch_size = 32, 
				  gamma = 0.99, 
				  update_interval = 1000, 
				  tau = 0.005,
				  eps_start = 0.9, 
				  eps_end = 0.05, 
				  eps_decay = 0.999999):
	
	episode_rewards = []
	eps = eps_start
	# Total steps played 
	total_steps = 0
	best_mean_reward = 1000.0
	
	model_no = 0
	for episode in range(max_episodes):
		for step in count():
			total_steps += 1
			eps = max(eps*eps_decay, eps_end)

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
					torch.save(agent.model.state_dict(), "Local_DQN_Mario_no" + str(model_no))
					print("Best mean reward updated, model saved")
					model_no += 1
					model_no = model_no % 10

				print("Steps: %d | Episodio: %d | mean reward %.3f | epsilon %.2f" % (total_steps, episode, mean_reward, eps))
				break
			
			optimizarModelo(agent, tau, batch_size, gamma, update_interval)

			# Guardar el modelo
			if total_steps % 250000 == 0:
				torch.save(agent.model.state_dict(), "Local_DQN_Mario_snapshot")
				print("snapshot saved")


		if best_mean_reward >= expected_reward:
			break


	torch.save(agent.model.state_dict(), "models/Local_DQN_Mario_snapshot_" + str(math.floor(time.time())))

	writer.close()
	return episode_rewards


# Ambiente
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SALTO_CORRER) #SIMPLE_MOVEMENT
env = wrap_env(env)
env.reset()


# hiperparametros
replay_start        = 10000		# steps minimos para empezar
replay_buffer_size  = 10000		# steps maximos que guarda 
BATCH_SIZE          = 32		# numero de experiencias del buffer a usar

TAU 				= 0.005		# actualizacion pasiva de la target network

mum_episodes		= 10000   
GAMMA				= 0.99
EPS_START			= 0.9
EPS_END				= 0.05
EPS_DECAY 			= 0.9999975
recompensa_termino	= 3000.0


writer = SummaryWriter('runs/' + str(math.floor(time.time())) + '-mario_1_1-' + str(EPS_DECAY) + '-' + str(replay_buffer_size))

agent = DQNAgent(env, replay_buffer_size, render=True)
#agent.load_params("Local_DQN_Mario_snapshot_1")

episode_rewards = entrenamiento(agent, 
							max_episodes	= mum_episodes, 
							writer			= writer, 
							expected_reward	= recompensa_termino, 
							replay_start	= replay_start, 
							batch_size		= BATCH_SIZE, 
							gamma			= GAMMA, 
							eps_start		= EPS_START, 
							eps_end			= EPS_END, 
							eps_decay		= EPS_DECAY,
							tau				= TAU)




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

grafico("DQN", episode_rewards)
