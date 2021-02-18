import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

from q_learning_mountaincar_rbf import plot_cost_to_go, FeatureTransformer, plot_running_avg

class BaseModel:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, input_, target, eligibility, lr=10e-2):
        self.w += lr*(target - input_.dot(self.w))*eligibility

    def predict(self, X):
        X = np.array(X)
        return X.dot(self.w)


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        D = feature_transformer.dimensions
        self.eligibilities = np.zeros((env.action_space.n, D))
        for i in range(env.action_space.n):
            self.models.append(BaseModel(D))

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        return np.array([m.predict(X)[0] for m in self.models])
    
    def update(self, s, a, G, gamma, lambda_):
        X = self.feature_transformer.transform([s])
        self.eligibilities *= gamma * lambda_
        self.eligibilities[a] += X[0]
        self.models[a].partial_fit(X[0], G, self.eligibilities[a])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def play_one(model, eps, gamma, lambda_):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        G = reward + gamma*np.max(model.predict(observation)[0])
        model.update(prev_observation, action, G, gamma, lambda_)

        totalreward += reward
        iters += 1
    return totalreward

if __name__ == '__main__':
    env = gym.make('MountainCar-v0') #.env
    ft = FeatureTransformer(env)
    model = Model(env, ft)

    gamma = 0.9999
    lambda_ = 0.7
    # filename = os.path.basename(__file__).split('.')[0]
    # monitor_dir = f'{filename}_{str(datetime.now())[:10]}'
    # env = wrappers.Monitor(env, monitor_dir)

    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one(model, eps, gamma, lambda_)
        totalrewards[n] = totalreward
        print(f'episode {n}, total reward {totalreward}')
    print(f'avg reward last hundo = {totalrewards[-100].mean()}')
    print(f'tot step = {-totalrewards.sum()}')

    plt.plot(totalrewards)
    plt.title('rewards')
    plt.show()

    plot_running_avg(totalrewards)
    plot_cost_to_go(env, model)