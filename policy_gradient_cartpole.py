import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from q_learning_bins import play_one, plot_running_avg

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.relu, use_bias=True):
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
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
    def __init__(self, D, K, hidden_layer_sizes):
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        layer = HiddenLayer(M1, K, tf.nn.softmax, use_bias=False)
        self.layers.append(layer)

        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        p_a_given_s = Z
        self.predict_op = p_a_given_s

        selected_probs = tf.log(
            tf.reduce_sum(
                p_a_given_s * tf.one_hot(self.actions, K),
                reduction_indices=[1]
            )
        )

        cost = -tf.reduce_sum(self.advantages * selected_probs)
        self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
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
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return np.random.choice(len(p), p=p)

class ValueModel:
    def __init__(self, D, hidden_layer_sizes):
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        layer = HiddenLayer(M1, 1, lambda x: x)
        self.layers.append(layer)

        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z, [-1])
        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

def play_one_td(env, pmodel, vmodel, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:
        action = pmodel.sample_action(observation)
        prev_obseravtion = observation
        observation, reward, done, info = env.step(action)

        v_next = vmodel.predict(observation)
        G = reward + gamma * np.max(v_next)
        advantage = G - vmodel.predict(prev_obseravtion)
        pmodel.partial_fit(prev_obseravtion, action, advantage)
        vmodel.partial_fit(prev_obseravtion, G)

        if reward == 1:
            totalreward += reward
        
        iters += 1
    return totalreward

def play_one_mc(env, pmodel, vmodel, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    states = []
    actions = []
    rewards = []

    while not done and iters < 2000:
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        if done:
            reward = -200

        states.append(prev_observation)
        actions.append(action)
        rewards.append(reward)

        if reward == 1:
            totalreward += reward
        iters += 1

    returns = []
    advantages = []
    G = 0
    for s, r in zip(states[::-1], actions[::-1]):
        returns.append(G)
        advantages.append(G - vmodel.predict(s)[0])
        G = r + gamma * G
    returns.reverse()
    advantages.reverse()

    pmodel.partial_fit(states, actions, advantages)
    vmodel.partial_fit(states, returns)

    return totalreward

def main():
    env = gym.make('CartPole-v0')
    D = env.observation_space.shape[0]
    K = env.action_space.n
    pmodel = PolicyModel(D, K, [4,4])
    vmodel = ValueModel(D, [4,4])
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.99

    N = 500
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        totalreward = play_one_mc(env, pmodel, vmodel, gamma)
        totalrewards[n] = totalreward
        print(f'episode {n}, total reward {totalreward}')
    print(f'avg reward last hundo = {totalrewards[-100].mean()}')
    print(f'tot step = {-totalrewards.sum()}')

    plt.plot(totalrewards)
    plt.title('rewards')
    plt.show()

    plot_running_avg(totalrewards)

if __name__ == "__main__":
    main()


