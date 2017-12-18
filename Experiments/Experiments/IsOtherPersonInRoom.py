import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import math
import json
import os
import sys
import ijson

def getObject(path):
    with open(path) as f:
        return np.array(json.load(f))

#https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def readTrainingData(dir, maxSize):

    positivesDirectory = os.path.join(dir, "positives")
    negativesDirectory = os.path.join(dir, "negatives")

    labels = []
    data = []

    i = 1
    files = os.listdir(positivesDirectory)[:maxSize]
    for file in files:
        print(str(i) + "/" + str(len(files)))
        data.append(getObject(os.path.join(positivesDirectory, file)))
        labels.append(np.array([1]))
        i = i + 1

    files = os.listdir(negativesDirectory)[:maxSize]
    i = 1
    for file in files:
        print(str(i) + "/" + str(len(files)))
        data.append(getObject(os.path.join(negativesDirectory, file)))
        labels.append(np.array([0]))
        i = i + 1

    dict = {}
    dict["DATA"] = np.array(data)
    dict["LABELS"] = np.array(labels)

    return dict

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
def max_pooling(x, poolSize):
    return tf.nn.max_pool(x, ksize=[1, poolSize, poolSize, 1], strides=[1, poolSize, poolSize, 1], padding="SAME")

batchSize = 4
frameWidth = 256
frameHeight = 212
frameSize = frameWidth * frameHeight
outputNeuronCount = 1
poolSize = 4

trainingData = readTrainingData("D:/cnnData/training_justjson", 100)

_x = tf.placeholder(dtype = tf.float32, shape = [batchSize, frameSize])
_labels = tf.placeholder(dtype=tf.int32, shape=[batchSize, outputNeuronCount])

W_conv1 = createWeight([25, 25, 1, 32])
b_conv1 = createBias([32])

reformatedImaged = tf.reshape(_x, [-1, frameWidth, frameHeight, 1])

conv1Handle = tf.nn.relu(conv2d(reformatedImaged, W_conv1) + b_conv1)
pool1Handle = max_pooling(conv1Handle, poolSize)

W_conv2 = createWeight([25, 25, 32, 64])
b_conv1 = createBias([64])

conv2Handle = tf.nn.relu(conv2d(pool1Handle, W_conv2) + b_conv1)
pool2Handle = max_pooling(conv2Handle, poolSize)

newWidth = int(math.ceil(frameWidth / (poolSize * poolSize)))
newHeight = int(math.ceil(frameHeight / (poolSize * poolSize)))

W_fc1 = createWeight([newHeight * newWidth * 64, 1024])
b_fc1 = createBias([1024])

pool2FlatLayer = tf.reshape(pool2Handle, [-1, newHeight * newWidth * 64])
fullyConnectedLayerHandle = tf.nn.relu(tf.matmul(pool2FlatLayer, W_fc1) + b_fc1)

probOfDropout = tf.placeholder(dtype=tf.float32)
handleDropoutLayer = tf.nn.dropout(fullyConnectedLayerHandle, probOfDropout)

W_output = createWeight([1024, 1])
b_output = createBias([1])

handleOfOutputLayer = tf.matmul(handleDropoutLayer, W_output) + b_output

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = _labels, logits = handleOfOutputLayer))
    trainer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    session.run(tf.initialize_all_variables())
    correctArray = tf.equal(tf.argmax(handleOfOutputLayer, 1), tf.argmax(_labels, 1))
    correctArrayAsInt = tf.to_float(correctArray)
    accuracy = tf.reduce_mean(correctArrayAsInt)

    for i in range(0, 20000):
        batch = next_batch(batchSize, trainingData["DATA"], trainingData["LABELS"])
        trainer.run(feed_dict = {_x: batch[0], _labels: batch[1], probOfDropout: .5})

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                _x: batch[0], _labels: batch[1], probOfDropout: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
