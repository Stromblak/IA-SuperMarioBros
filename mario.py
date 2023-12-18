import gym_super_mario_bros
from DDQN import wrap_env, DDQNAgent
import sys
import pygame

# DQN
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = wrap_env(env)
env.reset()


agent = DDQNAgent(env)

if sys.argv[1] == "1":
    agent.load_params("pesos1")
    nivel = "SuperMarioBros-1-1-v0"

elif sys.argv[1] == "4":
    agent.load_params("1702259439-[1,4]\DQN_G_2006.0")
    nivel = "SuperMarioBros-1-4-v0"    

pygame.init()
clock = pygame.time.Clock()
for episode in range(1, 5):
    env = gym_super_mario_bros.make(nivel)
    env = wrap_env(env)
    agent.state = env.reset()
    
    while True:
        
        clock.tick(30)

        env.render()
        if agent.playGlobal(0, env):
            break
    
    env.close()

        