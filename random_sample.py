import gym
import numpy as np
import matplotlib.pyplot as plt

def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, paramas, max_its=10000):
    observation = env.reset()
    done = False
    t = 0
    while not done and t < max_its:
        t += 1
        action = get_action(observation, paramas)
        observation, reward, done, info = env.step(action)
        if done:
            break
    return t

def play_multiple_episdoes(env, T, params):
    episode_lengths = np.empty(T)

    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)

    avg_length = episode_lengths.mean()
    print(f"average lenght: {avg_length}")

    return avg_length

def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4)*2 - 1
        avg_length = play_multiple_episdoes(env, 100, new_params)
        episode_lengths.append(avg_length)

        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params

if __name__ == "__main__":
    env = gym.make('CartPole-v0').env
    episode_lenghts, params = random_search(env)
    plt.plot(episode_lenghts)
    plt.show()

    print("final run")
    play_multiple_episdoes(env, 100, params)
    


