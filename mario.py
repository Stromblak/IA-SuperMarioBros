from gym_super_mario_bros import SuperMarioBrosEnv

env = SuperMarioBrosEnv()



def playgames(env, num_games, render = True):
    wins = 0
    env.reset()
    env.render()

    for i_episode in range(num_games):
        rewards_epi = 0
        t = 0

        while True:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            rewards_epi=rewards_epi+reward


            if render: 
                env.render()

            if done:
                if reward >= 0:
                    wins += 1
                print(f"Episode {i_episode} finished after {t+1} timesteps with reward {rewards_epi}")
                break
            
            t += 1


    env.close()
    print("Victorias: ", wins)


playgames(env, 1, True)