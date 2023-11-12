import numpy as np
import gym
import gym_gridworld
import matplotlib.pyplot as plt
env = gym.make("GridWorld-v0")

env.verbose = True
_ =env.reset()

EPISODES = 10000
MAX_STEPS = 1000

LEARNING_RATE = 0.05     # 0.05
GAMMA = 1
epsilon = 1             # 1
lmbda = 0.5

IMPRIMIR = 0        # para imprimir o no las recompensas por episodio

print('Observation space\n')
print(env.observation_space)


print('Action space\n')
print(env.action_space)


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
    
    #plt.savefig(nombre + ".png", format='png')


# metodos clasicos
def sarsa(env, epsilon):
    rewards = []
    rewards2 = []
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles
    
    Q = np.zeros((STATES, ACTIONS))
    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        
        if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.

        for actual_step in range(MAX_STEPS):

            next_state, reward, done, _ = env.step(action)
            
            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action2 = env.action_space.sample() 
            else:
                action2 = np.argmax(Q[next_state, :]) #De lo contrario, escogerá el estado con el mayor valor.

            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * Q[next_state, action2] - Q[state, action]) #Calcula la nueva Q table.
            rewards_epi=rewards_epi+reward
            state = next_state
            action = action2

            if (MAX_STEPS-2)<actual_step:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}") 
                if epsilon > 0.1: epsilon -= 0.0001

            if done:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}") 
                    print(f"Value of epsilon: {epsilon}")
                rewards.append(rewards_epi) #Guarda las recompensas en una lista
                if epsilon > 0.1: epsilon -= 0.0001
                break  

        rewards2.append(rewards_epi)
        

    print(Q)
    #print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    grafico("SARSA", rewards2)
    return Q

def qlearning(env, epsilon):
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles.

    Q = np.zeros((STATES, ACTIONS)) #Inicializa la Q table con 0s.
    rewards = []
    rewards2 = []
    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        for actual_step in range(MAX_STEPS):

            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action = env.action_space.sample() 
            else:
                action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.

            next_state, reward, done, _ = env.step(action) #Ejecuta la acción en el ambiente y guarda los nuevos parámetros (estado siguiente, recompensa, ¿terminó?).
            rewards_epi=rewards_epi+reward

            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]) #Calcula la nueva Q table.

            state = next_state

            if (MAX_STEPS-2)<actual_step:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}") 
                    print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1: epsilon -= 0.0001

            if done:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}") 
                    print(f"Value of epsilon: {epsilon}")
                rewards.append(rewards_epi) #Guarda las recompensas en una lista
                if epsilon > 0.1: epsilon -= 0.0001
                break  

        rewards2.append(rewards_epi)

    print(Q)

    grafico("Q-Learning", rewards2)


    #print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return Q


# Q learning doble
def qlearningDoble(env, epsilon):
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles.


    # Las dos tablas Q
    Q1 = np.zeros((STATES, ACTIONS))
    Q2 = np.zeros((STATES, ACTIONS))


    rewards = []
    rewards2 = []
    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        for actual_step in range(MAX_STEPS):

            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action = env.action_space.sample() 
            else:
                action = np.argmax(Q1[state, :] + Q2[state, :]) #De lo contrario, escogerá el estado con el mayor valor.

            next_state, reward, done, _ = env.step(action) #Ejecuta la acción en el ambiente y guarda los nuevos parámetros (estado siguiente, recompensa, ¿terminó?).
            rewards_epi=rewards_epi+reward

            # Actualizar de manera aleatoria una de las dos tablas Q
            if np.random.uniform(0, 1) < 0.5:
                Q1[state, action] = Q1[state, action] + LEARNING_RATE * (reward + GAMMA * Q2[next_state, np.argmax(Q1[next_state, :])] - Q1[state, action]) #Calcula la nueva Q1 table.
            
            else:
                Q2[state, action] = Q2[state, action] + LEARNING_RATE * (reward + GAMMA * Q1[next_state, np.argmax(Q2[next_state, :])] - Q2[state, action]) #Calcula la nueva Q2 table.


            state = next_state

            if (MAX_STEPS-2)<actual_step:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}") 
                    print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1: epsilon -= 0.0001

            if done:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}") 
                    print(f"Value of epsilon: {epsilon}")
                rewards.append(rewards_epi) #Guarda las recompensas en una lista
                if epsilon > 0.1: epsilon -= 0.0001
                break  

        rewards2.append(rewards_epi)
    
    print(Q1 + Q2)
    grafico("Q-Learning Doble", rewards2)

    #print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.

    # Retornar la suma de las dos tablas Q
    return Q1 + Q2



# TD lambda
def sarsaLambda(env, epsilon):
    rewards = []
    rewards2 = []
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles
    

    Q = np.zeros((STATES, ACTIONS)) # Tabla Q
    e = np.zeros((STATES, ACTIONS)) # Tabla eligibility traces


    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        

        # Inicializar s y a
        if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.



        for actual_step in range(MAX_STEPS):
            # Take action a ...
            next_state, reward, done, _ = env.step(action)
            
            # Choose a' ...
            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action2 = env.action_space.sample() 
            else:
                action2 = np.argmax(Q[next_state, :]) #De lo contrario, escogerá el estado con el mayor valor.


            delta = reward + GAMMA * Q[next_state, action2] - Q[state, action]
            e[state, action] = e[state, action] + 1


            # For all s, a:
            Q = Q + LEARNING_RATE * delta * e
            e = GAMMA * lmbda * e

            # actualizar s, a y recompensa
            rewards_epi=rewards_epi+reward
            state = next_state
            action = action2


            if (MAX_STEPS-2)<actual_step:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}") 
                if epsilon > 0.1: epsilon -= 0.0001

            if done:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}") 
                    print(f"Value of epsilon: {epsilon}")
                rewards.append(rewards_epi) #Guarda las recompensas en una lista
                if epsilon > 0.1: epsilon -= 0.0001
                break  

        rewards2.append(rewards_epi)
        

    print(Q)
    #print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    grafico("SARSA Lambda", rewards2)
    return Q

def qlambda(env, epsilon):
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles.

    Q = np.zeros((STATES, ACTIONS)) # Tabla Q
    e = np.zeros((STATES, ACTIONS)) # Tabla eligibility traces

    rewards = []
    rewards2 = []


    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        
        # Inicializar s, a
        if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.



        for actual_step in range(MAX_STEPS):
            # Take action a, observe r', s'
            next_state, reward, done, _ = env.step(action)
            
            # Choose a' from s' ...
            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action2 = env.action_space.sample() 
            else:
                action2 = np.argmax(Q[next_state, :]) #De lo contrario, escogerá el estado con el mayor valor.


            # a* = argmax( ...
            aAsterisco = np.argmax(Q[next_state, :])
            delta = reward + GAMMA * Q[next_state, aAsterisco] - Q[state, action]
            e[state, action] = e[state, action] + 1

            # for all s, a:
            Q = Q + LEARNING_RATE * delta * e
            if aAsterisco == action2: e = GAMMA * lmbda * e
            else: e = np.zeros((STATES, ACTIONS))
            
            # actualizar s, a y recompensa
            rewards_epi=rewards_epi+reward
            state = next_state
            action = action2

            if (MAX_STEPS-2)<actual_step:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}") 
                    print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1: epsilon -= 0.0001

            if done:
                if IMPRIMIR:
                    print (f"Episode {episode} rewards: {rewards_epi}") 
                    print(f"Value of epsilon: {epsilon}")
                rewards.append(rewards_epi) #Guarda las recompensas en una lista
                if epsilon > 0.1: epsilon -= 0.0001
                break  

        rewards2.append(rewards_epi)

    print(Q)

    grafico("Q Lambda", rewards2)


    #print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return Q




#Función para correr juegos siguiendo una determinada política
def playgames(env, Q, num_games, render = True):
    wins = 0
    env.reset()
    #pause=input()
    env.render()

    for i_episode in range(num_games):
        rewards_epi=0
        observation = env.reset()
        t = 0
        while True:
            action = np.argmax(Q[observation, :]) #La acción a realizar esta dada por la política
            observation, reward, done, info = env.step(action)
            rewards_epi=rewards_epi+reward
            if render: 
                env.render()
                import time
                time.sleep(0.01)
            if done:
                if reward >= 0:
                    wins += 1
                print(f"Episode {i_episode} finished after {t+1} timesteps with reward {rewards_epi}")
                break
            t += 1
    #pause=input()
    env.close()
    print("Victorias: ", wins)


Q = sarsa(env, epsilon)

#Q = qlearning(env, epsilon)

#Q = qlearningDoble(env, epsilon)

#Q = qlambda(env, epsilon)

#Q = sarsaLambda(env, epsilon)



plt.savefig("Grafico.png", format='png')

playgames(env, Q, 100, True)
env.close()



#_ =env.step(env.action_space.sample())
