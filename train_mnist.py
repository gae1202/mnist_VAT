import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('seed', 1, "initial random seed")
tf.app.flags.DEFINE_string('layer_sizes', '784-1200-600-300-150-10', "layer sizes")
tf.app.flags.DEFINE_integer('batch_size', 128, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('num_steps', 1000, "the number of epochs for training")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "initial leanring rate")


def inference(x):
    layer_sizes = np.asarray(FLAGS.layer_sizes.split('-'), np.int32)
    num_layers = len(layer_sizes) - 1
    h = x
    for layer_i, (in_units, out_units) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        W = tf.get_variable(
            name="W_"+str(layer_i),
            shape=[in_units, out_units],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(6.0 / (in_units+out_units))),
        )
        b = tf.get_variable(
            name="b_"+str(layer_i),
            shape=[out_units],
            initializer=tf.constant_initializer(0.0),
        )

        h = tf.matmul(h, W) + b
        if layer_i != (num_layers - 1):
            h = tf.nn.relu(h)
    return h


def loss_op(logits, t):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, t)
    return tf.reduce_mean(cross_entropy)


def accuracy_op(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def training(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

def main(_):
    rng = np.random.RandomState(FLAGS.seed)
    mnist = input_data.read_data_sets("data/", one_hot=True)
    graph = tf.Graph()
    with graph.as_default():
        layer_sizes = np.asarray(FLAGS.layer_sizes.split('-'), np.int32)
        x = tf.placeholder(tf.float32, [None, layer_sizes[0]])
        t = tf.placeholder(tf.float32, [None, layer_sizes[-1]])
        logits = inference(x)
        _loss = loss_op(logits, t)
        _accuracy = accuracy_op(logits, t)
        train_op = training(_loss)

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        for i in  xrange(FLAGS.num_steps):
            batch_xs, batch_ts = mnist.train.next_batch(FLAGS.batch_size)
            feed_dict = {
                x: batch_xs,
                t: batch_ts,
            }
            loss, accuracy, _ = sess.run([_loss, _accuracy, train_op], feed_dict=feed_dict)
            step = i + 1
            if (step % 100 == 0) or step == FLAGS.num_steps:
                print("[Step {0}]  loss:{1}, accuracy:{2}".format(step, loss, accuracy))

if __name__ == "__main__":
    tf.app.run()