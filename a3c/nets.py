from numpy.core.fromnumeric import squeeze
import tensorflow as tf

def build_feature_extractor(input_):
    input_ = tf.to_float(input_) / 255.0

    conv1 = tf.contrib.layers.conv2d(
        input_, 16, 8, 4,
        activation_fn=tf.nn.relu,
        scope='conv1'
    )
    conv2 = tf.contrib.layers.conv2d(
        conv1, 32, 4, 2,
        activation_fn=tf.nn.relu,
        scope='conv2'
    )
    flat = tf.contrib.layers.flatten(conv2)

    fc1 = tf.contrib.layers.fully_connected(
        inputs=flat,
        num_outputs=256,
        scope='fc1'
    )
    return fc1


class PolicyNetwork:
    def __init__(self, num_outputs, reg=0.01):
        self.num_outputs = num_outputs
        self.states = tf.placeholder(shape=(None, 84, 84, 4), dtype=tf.uint8, name='X')
        self.advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name='y')
        self.actions = tf.placeholder(shape=(None,), dtype=tf.int32, name='actions')

        with tf.variable_scope("shared", reuse=False):
            fc1 = build_feature_extractor(self.states)

        with tf.variable_scope("policy_network"):
            self.logits = tf.contrib.layers.fully_connected(
                fc1, num_outputs, activation_fn=None
            )
            self.probs = tf.nn.softmax(self.logits)

            cdist = tf.distributions.Categorical(logits=self.logits)
            self.sample_action = cdist.sample()

            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), axis=1)

            batch_size = tf.shape(self.states)[0]
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.selected_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

            self.loss = tf.log(self.selected_action_probs) * self.advantage + reg * self. entropy
            self.loss = - tf.reduce_sum(self.loss, name='loss')

            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]

        
class ValueNetwork:
    def __init__(self):

        self.states = tf.placeholder(shape=(None, 84, 84, 4), dtype=tf.uint8, name="X")
        self.targets = tf.placeholder(shape=(None,), dtype=tf.float32, name="y")

        with tf.variable_scope("shared", reuse=True):
            fc1 =build_feature_extractor(self.states)

        with tf.variable_scope("value_network"):
            self.vhat = tf.contrib.layers.fully_connected(
                inputs=fc1, num_outputs=1,
                activation_fn=None
            )
            self.vhat = tf.squeeze(self.vhat, squeeze_dims=[1], name="vhat")

            self.loss = tf.squared_difference(self.vhat, self.targets)
            self.loss = tf.reduce_sum(self.loss, name='loss')

            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]


def create_networks(num_outputs):
    policy_network = PolicyNetwork(num_outputs=num_outputs)
    value_network = ValueNetwork()
    return policy_network, value_network