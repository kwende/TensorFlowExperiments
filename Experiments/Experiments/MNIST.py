import tensorflow as tf
import math

IMAGE_DIM = 28
IMAGE_SIZE = IMAGE_DIM * IMAGE_DIM

hidden1_size = 100
hidden2_size = 30
output_size = 10

imagePixels = tf.zeros([1, IMAGE_SIZE])

# input layer to first hidden.
hidden1Weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE, hidden1_size], stddev=1.0 / math.sqrt(IMAGE_SIZE)))
hidden1Biases = tf.Variable(tf.zeros(hidden1_size))
hidden1 = tf.nn.relu(tf.matmul(imagePixels, hidden1Weights) + hidden1Biases)

# first hidden to second hidden.
hidden2Weights = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=1.0 / math.sqrt(hidden1_size)))
hidden2Biases = tf.Variable(tf.zeros(hidden2_size))
hidden2 = tf.nn.relu(tf.matmul(hidden1Weights, hidden2Weights) + hidden2Biases)

# second hidden to output layer.
outputWeights = tf.Variable(tf.truncated_normal([hidden2_size, output_size], stddev=1.0 / math.sqrt(output_size)))
outputBias = tf.Variable(tf.zeros(output_size))
output = tf.matmul(hidden2Weights, outputWeights) + outputBias

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)

