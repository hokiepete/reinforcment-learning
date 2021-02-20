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
            self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)

class PolicyModel:
    def __init__(self, ft, D, hidden_layer_sizes_mean=tuple(), hidden_layer_sizes_var=tuple()):
        
        self.ft = ft
        self.D = D
        self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
        self.hidden_layer_sizes_var = hidden_layer_sizes_var
        
        self.mean_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_mean:
            layer = HiddenLayer(M1, M2)
            self.mean_layers.append(layer)
            M1 = M2
        
        layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
        self.mean_layers.append(layer)


        self.var_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_var:
            layer = HiddenLayer(M1, M2)
            self.var_layers.append(layer)
            M1 = M2
        
        layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)
        self.var_layers.append(layer)


        self.params = []
        for layer in (self.mean_layers + self.var_layers):
            self.params += layer.params

        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        def get_output(layers):
            Z = self.X
            for layer in layers:
                Z = layer.forward(Z)
            return tf.reshape(Z, [-1])

        mean = get_output(self.mean_layers)
        var = get_output(self.var_layers) + 10e-5

        norm = tf.contrib.distributions.Normal(mean, var)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

        # selected_probs = tf.log(
        #     tf.reduce_sum(
        #         p_a_given_s * tf.one_hot(self.actions, K),
        #         reduction_indices=[1]
        #     )
        # )

        # cost = -tf.reduce_sum(self.advantages * selected_probs)
        # self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)

    def set_session(self, session):
        self.session = session

    def init_vars(self):
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)

    # def partial_fit(self, X, actions, advantages):
    #     X = np.atleast_2d(X)
    #     actions = np.atleast_1d(actions)
    #     advantages = np.atleast_1d(advantages)
    #     self.session.run(
    #         self.train_op,
    #         feed_dict={
    #             self.X: X,
    #             self.actions: actions,
    #             self.advantages: advantages,
    #         }
    #     )

    def predict(self, X):
        X = self.ft.transform(np.atleast_2d(X))
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        # return np.random.choice(len(p), p=p)
        return p

    def copy(self):
        clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var)
        clone.set_session(self.session)
        clone.init_vars()
        clone.copy_from(self)
        return clone

    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def perturb_params(self):
        ops = []
        for p in self.params:
            v = self.session.run(p)
            noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
            if np.random.random() < 0.1:
                op = p.assign(noise)
            else:
                op = p.assign(v + noise)
            ops.append(op)
        self.session.run(ops)



def play_one(env, pmodel, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:
        action = pmodel.sample_action(observation)
        observation, reward, done, info = env.step([action])
        totalreward += reward
        iters += 1
    return totalreward

def play_multiple_episodes(env, T, pmodel, gamma, print_iters=False):
    totalrewards = np.empty(T)
    
    for i in range(T):
        totalrewards[i] = play_one(env, pmodel, gamma)
    
    avg_totalrewards = totalrewards.mean()
    print(f"avg totalrewards: {avg_totalrewards}")
    return avg_totalrewards

def random_search(env, pmodel, gamma):
    totalrewards = []
    best_avg_totalreward = float('-inf')
    best_pmodel = pmodel
    num_episodes_per_param_test = 3
    for t in range(100):
        tmp_pmodel = best_pmodel.copy()
        tmp_pmodel.perturb_params()

        avg_totalrewards = play_multiple_episodes(
            env,
            num_episodes_per_param_test,
            tmp_pmodel,
            gamma
        )
        totalrewards.append(avg_totalrewards)

        if avg_totalrewards > best_avg_totalreward:
            best_pmodel = tmp_pmodel
            best_avg_totalreward = avg_totalrewards
    return totalrewards, best_pmodel

def main():
    env = gym.make('MountainCarContinuous-v0').env
    ft = FeatureTransformer(env, n_components=100)
    D = ft.dimensions
    pmodel = PolicyModel(ft, D, [], [])
    session = tf.InteractiveSession()
    pmodel.set_session(session)
    pmodel.init_vars()
    gamma = 0.99

    
    totalreward, pmodel = random_search(
        env, pmodel, gamma
    )
    print(f'max rewards {max(totalreward)}')
    
    avg_totalrewards = play_multiple_episodes(env, 100, pmodel, gamma)
    print(f'avg reward = {avg_totalrewards}')
    
    plt.plot(totalreward)
    plt.title('rewards')
    plt.show()

    plot_running_avg(totalreward)

if __name__ == "__main__":
    main()


