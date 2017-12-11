import tensorflow as tf
import math
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

IMAGE_DIM = 28
IMAGE_SIZE = IMAGE_DIM * IMAGE_DIM

hidden1_size = 100
hidden2_size = 30
output_size = 10
batch_size = 1

imagePixelsPlaceholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784])
labelsPlaceholder = tf.placeholder(dtype=tf.int32, shape=[batch_size])

# input layer to first hidden.
hidden1Weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE, hidden1_size], stddev=1.0 / math.sqrt(IMAGE_SIZE)))
hidden1Biases = tf.Variable(tf.zeros(hidden1_size))
hidden1 = tf.nn.relu(tf.matmul(imagePixelsPlaceholder, hidden1Weights) + hidden1Biases)

# first hidden to second hidden.
hidden2Weights = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=1.0 / math.sqrt(hidden1_size)))
hidden2Biases = tf.Variable(tf.zeros(hidden2_size))
hidden2 = tf.nn.relu(tf.matmul(hidden1, hidden2Weights) + hidden2Biases)

# second hidden to output layer.
outputWeights = tf.Variable(tf.truncated_normal([hidden2_size, output_size], stddev=1.0 / math.sqrt(output_size)))
outputBias = tf.Variable(tf.zeros(output_size))
output = tf.matmul(hidden2, outputWeights) + outputBias

# identify the loss function. 
crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(labelsPlaceholder), logits=output)
lossFunction = tf.reduce_mean(crossEntropy)

# create the learner (or optimizer) against the loss function. 
optimizer = tf.train.GradientDescentOptimizer(.01)
trainer = optimizer.minimize(lossFunction)

# initialize the variables. 
init = tf.global_variables_initializer()

mnistData = input_data.read_data_sets("MNIST_data/")

pixels = np.reshape(mnistData.train.images[0], [1, 784])
labels =  [mnistData.train.labels[0]]

session = tf.Session()
session.run(init)

for step in range(0, 1000):
    ret = session.run([trainer, lossFunction], feed_dict = {imagePixelsPlaceholder:pixels, labelsPlaceholder:labels})
    print(ret[1])


#pixels = np.reshape(mnistData.train.images[0], [1, 784])
#labels =  [mnistData.train.labels[0]]

#print(session.run(output, {imagePixelsPlaceholder : pixels,
#                     labelsPlaceholder : tf.to_int64(labels), }))

print()

