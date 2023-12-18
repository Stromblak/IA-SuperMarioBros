from DDQN import *
import torch.nn.init as init

POBLACION = 10  

def encontrar_indices_extremos(lista):
    # Enumerar la lista para obtener pares de (índice, valor)
    pares = list(enumerate(lista))

    # Ordenar la lista de pares por el valor (en orden ascendente)
    pares_ordenados = sorted(pares, key=lambda x: x[1])

    # Obtener los dos índices con los valores más bajos
    indices_valores_bajos = [pares_ordenados[i][0] for i in range(2)]

    # Obtener los dos índices con los valores más altos
    indices_valores_altos = [pares_ordenados[-i-1][0] for i in range(2)]

    return indices_valores_bajos, indices_valores_altos


def init_weights(model):
	for m in model.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				init.constant_(m.bias.data, 0)

	return model


def genetico():

	# inicio
	env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
	env = wrap_env(env)
	env.reset()

	marios = []
	fitness = []

	for i in range(POBLACION):
		marios.append(DDQNAgent(env, render=False))
		marios[i].policy = init_weights(marios[i].policy)
		marios[i].target = init_weights(marios[i].target)
		fitness.append(float('-inf'))

	carpeta = str(math.floor(time.time())) + "/"
	writer = SummaryWriter(carpeta)


	# bucle
	for episode in range(0):

		# fitness
		for i in range(POBLACION):
			marios[i].state = env.reset()

			while True:
				reward = marios[i].playGlobal(0, env)

				if reward is not None:
					fitness[i] = reward
					break

		
		# desendencia
		indices_bajos, indices_altos = encontrar_indices_extremos(fitness)

		

	exit()



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


genetico()