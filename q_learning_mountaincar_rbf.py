import gym
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

class FeatureTransformer:
    def __init__(self, env):
        observation_examples = np.array([
            env.observation_space.sample() for x in range(10000)
        ])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=500)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=500)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=500)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=500))
        ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))
        
        self.scaler = scaler
        self.featurizer = featurizer
        self.dimensions = example_features.shape[1]

    def transform(self, obervations):
        scaled = self.scaler.transform(obervations)
        return self.featurizer.transform(scaled)

class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(
                feature_transformer.transform([env.reset()]), [0]
            )
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        G = reward + gamma*np.max(model.predict(observation)[0])
        model.update(prev_observation, action, G)

        totalreward += reward
        iters += 1
    return totalreward

def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0],
        num=num_tiles
    )
    y = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0],
        num=num_tiles
    )
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(
        lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y])
    )
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm
    )
    ax.set_xlabel('pos')
    ax.set_ylabel('vel')
    ax.set_zlabel('cost')
    # ax.title("cost to go")
    fig.colorbar(surf)
    plt.show()

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title('run avg')
    plt.show()

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