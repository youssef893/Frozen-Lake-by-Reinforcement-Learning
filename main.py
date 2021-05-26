import gym
import numpy as np
from IPython.display import clear_output
import time
from Q_Learning import FrozenLakeAgent

env = gym.make('FrozenLake-v0')

agent = FrozenLakeAgent(env, 500_000, 200, 0.01, 0.9, 1.0, 0.01)


def play(agent, num_episodes):
    """Let the agent play Frozen Lake"""

    time.sleep(2)
    counter = 0
    for episode in range(num_episodes):
        state = agent.env.reset()
        done = False
        print('❄️🕳🥶 Frozen Lake - Episode ', episode + 1, '⛸🥏🏆 \n\n\n\n')

        time.sleep(1.5)

        steps = 0

        while not done:
            clear_output(wait=True)
            agent.env.render()
            time.sleep(0.3)

            action = np.argmax(agent.q_table[state])
            state, reward, done, _ = agent.env.step(action)
            steps += 1

        clear_output(wait=True)
        agent.env.render()

        if reward == 1:
            print(f'Yay! 🏆 You have found your 🥏 in {steps} steps. round{episode}')
            counter += 1
            time.sleep(2)
        else:
            print('Oooops 🥶 you fell through a 🕳, try again!')
            time.sleep(2)
        clear_output(wait=True)
    print("accuracy =", (counter / num_episodes) * 100, "%")


agent.train()
agent.plot()
play(agent, 10)
