import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime

import q_learning_mountaincar_rbf
from q_learning_mountaincar_rbf import plot_running_avg, FeatureTransformer, Model, plot_cost_to_go

class SGDRegressor:
    def __init__(self, **kwargs):
        self.w = None
        self.lr = 10e-3

    def partial_fit(self, X, Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)
    
q_learning_mountaincar_rbf.SGDRegressor = SGDRegressor

def play_one(model, eps, gamma, n=5):
    observation = env.reset()
    done = False
    totalreward = 0
    rewards = []
    states = []
    actions = []
    iters = 0

    multipier = np.array([gamma]*n)**np.arange(n)

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        
        states.append(observation)
        actions.append(action)

        prev_obseravtion = observation
        observation, reward, done, info = env.step(action)

        rewards.append(reward)

        if len(rewards) >= n:
            return_up_to_prediction = multipier.dot(rewards[-n:])
            G = return_up_to_prediction \
                + (gamma**n)* np.max(
                    model.predict(observation)[0]
                )
            model.update(states[-n], actions[-n], G)
        
        totalreward += reward
        iters += 1

    rewards = rewards[-n+1:]
    states = states[-n+1:]
    actions = actions[-n+1:]

    if observation[0] >= 0.5:
        while len(rewards) > 0:
            G = multipier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    else:
        while len(rewards) > 0:
            guess_rewards = rewards + [-1]*(n - len(rewards))
            G = multipier.dot(guess_rewards)
            model.update(states[0],actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    return totalreward

if __name__ == '__main__':
    env = gym.make('MountainCar-v0') #.env
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')

    gamma = 0.99

    # filename = os.path.basename(__file__).split('.')[0]
    # monitor_dir = f'{filename}_{str(datetime.now())[:10]}'
    # env = wrappers.Monitor(env, monitor_dir)

    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        print(f'episode {n}, total reward {totalreward}')
    print(f'avg reward last hundo = {totalrewards[-100].mean()}')
    print(f'tot step = {-totalrewards.sum()}')

    plt.plot(totalrewards)
    plt.title('rewards')
    plt.show()

    plot_running_avg(totalrewards)
    plot_cost_to_go(env, model)
