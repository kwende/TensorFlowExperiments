import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import math

# create a weight that's a normal distribution about 0 w/ stdev of .1
def createWeight(shape):
    weight = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(weight)

# create a positively valued bias to keep it from hitting the zero gradient
# (relu)
def createBias(shape):
    bias = tf.constant(.1, shape=shape)
    return tf.Variable(bias)

# build a convolution layer.  Zero pad for same size.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

# max pooling.
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

batchSize = 100

_x = tf.placeholder(dtype = tf.float32, shape = [batchSize, 784])
_labels = tf.placeholder(dtype=tf.int32, shape=[batchSize, 10])

W_conv1 = createWeight([5, 5, 1, 32])
b_conv1 = createBias([32])

reformatedImaged = tf.reshape(_x, [-1, 28, 28, 1])

conv1Handle = tf.nn.relu(conv2d(reformatedImaged, W_conv1) + b_conv1)
pool1Handle = max_pool_2x2(conv1Handle)

W_conv2 = createWeight([5, 5, 32, 64])
b_conv1 = createBias([64])

conv2Handle = tf.nn.relu(conv2d(pool1Handle, W_conv2) + b_conv1)
pool2Handle = max_pool_2x2(conv2Handle)

W_fc1 = createWeight([7 * 7 * 64, 1024])
b_fc1 = createBias([1024])

pool2FlatLayer = tf.reshape(pool2Handle, [-1, 7 * 7 * 64])
fullyConnectedLayerHandle = tf.nn.relu(tf.matmul(pool2FlatLayer, W_fc1) + b_fc1)

probOfDropout = tf.placeholder(dtype=tf.float32)
handleDropoutLayer = tf.nn.dropout(fullyConnectedLayerHandle, probOfDropout)

W_output = createWeight([1024, 10])
b_output = createBias([10])

handleOfOutputLayer = tf.matmul(handleDropoutLayer, W_output) + b_output

mnistData = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = _labels, logits = handleOfOutputLayer))
    trainer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    session.run(tf.initialize_all_variables())
    correctArray = tf.equal(tf.argmax(handleOfOutputLayer, 1), tf.argmax(_labels, 1))
    correctArrayAsInt = tf.to_float(correctArray)
    accuracy = tf.reduce_mean(correctArrayAsInt)

    for i in range(0, 20000):
        batch = mnistData.train.next_batch(batchSize)
        trainer.run(feed_dict = {_x: batch[0], _labels: batch[1], probOfDropout: .5})

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                _x: batch[0], _labels: batch[1], probOfDropout: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))

