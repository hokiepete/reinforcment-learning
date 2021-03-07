import os
import sys
import gym
import numpy as np
import tensorflow as tf

from nets import create_networks

class Step:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ImageTransformer:
    def __init__(self):
        with tf.variable_scope('image_transformer'):
            self.input_state = tf.placeholder(shape=(210, 160, 3), dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, (84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            self.output = tf.squeeze(self.output)

    def transform(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output, {self.input_state: state})


def repeat_frame(frame):
    return np.stack([frame] * 4, axis=2)


def shift_frames(state, next_frame):
    return np.append(state[:,:, 1:], np.expand_dims(next_frame, 2), axis=2)


def get_copy_params_op(src_vars, dst_vars):
    src_vars = list(sorted(src_vars, key=lambda v: v.name))
    dst_vars = list(sorted(dst_vars, key=lambda v: v.name))

    ops = []
    for s, d in zip(src_vars, dst_vars):
        op = d.assign(s)
        ops.append(op)
    return ops


def make_train_op(local_net, global_net):

    local_grads, _ = zip(*local_net.grads_and_vars)

    local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)

    _, global_vars = zip(*global_net.grads_and_vars)

    local_grads_global_vars = list(zip(local_grads, global_vars))

    return global_net.optimizer.apply_gradients(
        local_grads_global_vars, global_step=tf.train.get_global_step()
    )


class Worker:
    def __init__(
        self,
        name, 
        env, 
        policy_net, 
        value_net, 
        global_counter,
        returns_list, 
        discount_factor=0.99, 
        max_global_steps=None
    ):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.train.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.img_transformer = ImageTransformer()
        self.env = env

        with tf.variable_scope(name):
            self.policy_net, self.value_net = create_networks(policy_net.num_outputs)

        self.copy_params_op = get_copy_params_op(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global'),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/')
        )

        self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
        self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

        self.state = None
        self.total_reward = 0
        self.returns_list = returns_list

    def run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            self.state = repeat_frame(self.img_transformer.transform(self.env.reset()))

            try:
                while not coord.should_stop():
                    sess.run(self.copy_params_op)

                    steps, global_step = self.run_n_steps(t_max, sess)

                    if self.max_global_steps is not None and global_step >= self.max_global_steps:
                        coord.request_stop()
                        return
                    
                    self.update(steps, sess)

            except tf.errors.CancelledError:
                return

    def sample_action(self, state, sess):
        feed_dict = {self.policy_net.states: [state]}
        actions = sess.run(self.policy_net.sample_action, feed_dict)
        return actions[0]

    def get_value_prediction(self, state, sess):
        feed_dict = {self.value_net.states: [state]}
        vhat = sess.run(self.value_net.vhat, feed_dict)
        return vhat[0]

    def run_n_steps(self, n, sess):
        steps = []
        for _ in range(n):
            action = self.sample_action(self.state, sess)
            next_frame, reward, done, _ = self.env.step(action)

            next_state = shift_frames(self.state, self.img_transformer.transform(next_frame))

            if done:
                print(f"Total reward: {self.total_reward}, Worker: {self.name}")
                self.returns_list.append(self.total_reward)
                if len(self.returns_list) > 0 and len(self.returns_list) % 100 == 0:
                    print(f"*** Total average reward (last 100: {np.mean(self.returns_list[-100:])}, Collected so far: {len(self.returns_list)}")
                self.total_reward = 0
            else:
                self.total_reward += reward

            step = Step(self.state, action, reward, next_state, done)
            steps.append(step)

            global_step = next(self.global_counter)

            if done:
                self.state = repeat_frame(self.img_transformer.transform(self.env.reset()))
                break
            else:
                self.state = next_state
        return steps, global_step

    def update(self, steps, sess):
        
        reward = 0.0
        if not steps[-1].done:
            reward = self.get_value_prediction(steps[-1].next_state, sess)

        states = []
        advantages = []
        value_targets = []
        actions = []

        for step in reversed(steps):
            reward = step.reward + self.discount_factor * reward
            advantage = reward - self.get_value_prediction(step.state, sess)

            states.append(step.state)
            actions.append(step.action)
            advantages.append(advantage)
            value_targets.append(reward)

        feed_dict = {
            self.policy_net.states: np.array(states),
            self.policy_net.advantage: advantages,
            self.policy_net.actions: actions,
            self.value_net.states: np.array(states),
            self.value_net.targets: value_targets
        }

        global_step, pnet_loss, vnet_loss, _, _ = sess.run([
            self.global_step,
            self.policy_net.loss,
            self.value_net.loss,
            self.pnet_train_op,
            self.vnet_train_op
        ], feed_dict)
                
        return pnet_loss, vnet_loss


        
        