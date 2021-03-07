import gym
import sys
import os
import numpy as np
from numpy.testing._private.utils import nulp_diff
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import shutil
import threading
import multiprocessing

from nets import create_networks
from worker import Worker

ENV_NAME = "Breakout-v0"
MAX_GLOBAL_STEPS = 5e6
STEPS_PER_UPDATE = 5

def Env():
    return gym.envs.make(ENV_NAME)

if ENV_NAME == "Breakout-v0" or ENV_NAME == "Pong-v0":
    NUM_ACTIONS = 4
else:
    env = Env()
    NUM_ACTIONS = env.action_space.n
    env.close()

def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)


NUM_WORKERS = multiprocessing.cpu_count()

with tf.device("/cpu:0"):

    global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.variable_scope("global") as vs:
        policy_net, value_net = create_networks(NUM_ACTIONS)
    
    global_counter = itertools.count()

    returns_list = []

    workers = []
    for worker_id in range(NUM_WORKERS):
        worker = Worker(
            name=f"worker_{worker_id}",
            env=Env(),
            policy_net=policy_net,
            value_net=value_net,
            global_counter=global_counter,
            returns_list=returns_list,
            discount_factor=0.99,
            max_global_steps=MAX_GLOBAL_STEPS
        )
        workers.append(worker)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        worker_threads = []
        for worker in workers:
            worker_fn = lambda: worker.run(sess, coord, STEPS_PER_UPDATE)
            t = threading.Thread(target=worker_fn)
            t.start()
            worker_threads.append(t)

        coord.join(worker_threads, stop_grace_period_secs=300)

        x = np.array(returns_list)
        y = smooth(x)
        plt.plot(x, label='orig')
        plt.plot(y, label='smoothed')
        plt.legend()
        plt.show()