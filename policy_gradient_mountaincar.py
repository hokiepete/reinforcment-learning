import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

from q_learning_bins import play_one, plot_running_avg
from q_learning_mountaincar_rbf import plot_cost_to_go

class FeatureTransformer:
    def __init__(self, env, n_components=500):
        observation_examples = np.array([
            env.observation_space.sample() for x in range(10000)
        ])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=n_components)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=n_components)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=n_components)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=n_components))
        ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))
        
        self.scaler = scaler
        self.featurizer = featurizer
        self.dimensions = example_features.shape[1]
    
    def transform(self, obervations):
        scaled = self.scaler.transform(obervations)
        return self.featurizer.transform(scaled)

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
        if zeros:
            W = np.zeros((M1, M2)).astype(np.float32)
            self.W = tf.Variable(W)
        else:
            W = tf.random_normal(shape=(M1, M2)) * np.sqrt(2.0 / M1, dtype=np.float32)
            self.W = tf.Variable(W)
        
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)

class PolicyModel:
    def __init__(self, ft, D, hidden_layer_sizes=tuple()):
        
        self.ft = ft
        
        self.hidden_layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.hidden_layers.append(layer)
            M1 = M2
        
        self.mean_layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
        self.var_layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)

        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        Z = self.X
        for layer in self.hidden_layers:
            Z = layer.forward(Z)
        mean = self.mean_layer.forward(Z)
        var = self.var_layer.forward(Z) + 10e-5

        mean = tf.reshape(mean, [-1])
        var = tf.reshape(var, [-1])

        norm = tf.contrib.distributions.Normal(mean, var)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

        log_probs = norm.log_prob(self.actions)

        cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*norm.entropy())
        self.train_op = tf.train.AdagradOptimizer(1e-3).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(
            self.train_op,
            feed_dict={
                self.X: X,
                self.actions: actions,
                self.advantages: advantages,
            }
        )

    def predict(self, X):
        X = self.ft.transform(np.atleast_2d(X))
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        # return np.random.choice(len(p), p=p)
        return p


class ValueModel:
    def __init__(self, ft, D, hidden_layer_sizes=tuple()):
        self.ft = ft
        self.costs = []

        self.hidden_layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.hidden_layers.append(layer)
            M1 = M2
        
        layer = HiddenLayer(M1, 1, lambda x: x)
        self.hidden_layers.append(layer)

        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
        
        Z = self.X
        for layer in self.hidden_layers:
            Z = layer.forward(Z)
        
        Y_hat = tf.reshape(Z, [-1])
        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.cost = cost
        self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        Y = np.atleast_1d(Y)
        self.session.run(
            self.train_op,
            feed_dict={
                self.X: X,
                self.Y: Y
            }
        )
        cost = self.session.run(
            self.cost,
            feed_dict={
                self.X: X,
                self.Y: Y
            }
        )
        self.costs.append(cost)

    def predict(self, X):
        X = self.ft.transform(np.atleast_2d(X))
        return self.session.run(self.predict_op, feed_dict={self.X: X})


def play_one_td(env, pmodel, vmodel, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:
        action = pmodel.sample_action(observation)
        prev_obseraction = observation
        observation, reward, done, info = env.step([action])
        
        totalreward += reward
        
        v_next = vmodel.predict(observation)
        G = reward + gamma*v_next
        advantage = G - vmodel.predict(prev_obseraction)
        pmodel.partial_fit(prev_obseraction, action, advantage)
        vmodel.partial_fit(prev_obseraction, G)

        iters += 1

    return totalreward, iters

def play_multiple_episodes(env, T, pmodel, gamma, print_iters=False):
    totalrewards = np.empty(T)
    
    for i in range(T):
        totalrewards[i] = play_one(env, pmodel, gamma)
    
    avg_totalrewards = totalrewards.mean()
    print(f"avg totalrewards: {avg_totalrewards}")
    return avg_totalrewards

def main():
    env = gym.make('MountainCarContinuous-v0').env
    ft = FeatureTransformer(env, n_components=100)
    D = ft.dimensions
    pmodel = PolicyModel(ft, D, [])
    vmodel = ValueModel(ft, D, [])
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.95

    N = 50
    totalrewards = np.empty(N)
    for n in range(N):
        totalreward, _ = play_one_td(env, pmodel, vmodel, gamma)
        totalrewards[n] = totalreward

        print(f"episode {n}, total rewards {totalreward}")

    
    plt.plot(totalrewards)
    plt.title('rewards')
    plt.show()

    plot_running_avg(totalrewards)
    plot_cost_to_go(env, vmodel)

if __name__ == "__main__":
    main()


