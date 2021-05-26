import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


class FrozenLakeAgent:
    def __init__(self, env, num_episodes, max_steps, learning_rate,
                 gamma, epsilon, decay_rate):
        self.env = env
        state_size = self.env.observation_space.n
        action_num = self.env.action_space.n
        self.q_table = np.zeros((state_size, action_num))
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.average_reward = []

    def test_matrix(self, q_table, episode):
        total_reward = 0
        for i in range(100):
            s = self.env.reset()
            done = False
            while not done:
                a = np.argmax(q_table[s])
                s, r, done, _ = self.env.step(a)
                total_reward += r

        result = total_reward / 100
        print('Episode: {:,}, Average reward: {}'.format(episode, result))
        return result

    def update_q_table(self, state, action):
        new_state, reward, done, _ = self.env.step(action)
        self.q_table[state, action] = self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.gamma * np.max(self.q_table[new_state]
                                                                                         - self.q_table[state, action]))

        return new_state, reward, done

    def epsilon_greedy(self, state):
        """
        Returns the next action by exploration with probability epsilon
        and exploitation with probability 1-epsilon.
        """
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def decay_epsilon(self, episode):
        """
        Decaying exploration with the number of episodes.
        """
        self.epsilon = 0.1 + 0.9 * np.exp(-self.decay_rate * episode)

    def train(self):
        self.average_reward = []
        self.episode_length = np.zeros(self.num_episodes)

        for episode in range(self.num_episodes):
            state = self.env.reset()

            for step in range(self.max_steps):
                action = self.epsilon_greedy(state)
                state, reward, done = self.update_q_table(state, action)

                self.episode_length[episode] += 1

                if done:
                    break

            self.decay_epsilon(episode)
            if episode % 1000 == 0:
                avg_reward = self.test_matrix(self.q_table, episode)
                self.average_reward.append(avg_reward)
                if avg_reward > 0.8:
                    # considered "solved" when the agent get an avg of at least
                    # 0.78 over 100 in a row.
                    print("Frozen Lake solved 🏆🏆🏆")
                    break

    def plot(self):
        """Plot the episode length and average rewards per episode"""

        fig = plt.figure(figsize=(20, 5))

        episode_len = [i for i in self.episode_length if i != 0]

        rolling_len = pd.DataFrame(episode_len).rolling(100, min_periods=100)
        mean_len = rolling_len.mean()
        std_len = rolling_len.std()

        plt.plot(mean_len, color='red')
        plt.fill_between(x=std_len.index, y1=(mean_len - std_len)[0],
                         y2=(mean_len + std_len)[0], color='red', alpha=.2)

        plt.ylabel('Episode length')
        plt.xlabel('Episode')
        plt.title(
            f'Frozen Lake - Length of episodes (mean over window size 100)')
        plt.show()

        fig = plt.figure(figsize=(20, 5))

        plt.plot(self.average_reward, color='red')
        plt.gca().set_xticklabels(
            [i + i * 999 for i in range(len(self.average_reward))])

        plt.ylabel('Average Reward')
        plt.xlabel('Episode')
        plt.title(f'Frozen Lake - Average rewards per episode ')
        plt.show()


