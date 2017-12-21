import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import math
import json
import os
import sys
import ijson
import pickle

def getObject(path):
    with open(path) as f:
        return np.array(json.load(f))
        #return np.array([b/10000.0 for b in json.load(f)])

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

def readTrainingData(dir, maxSize, percentageOfDataIsValidation):

    positivesDirectory = os.path.join(dir, "positives")
    negativesDirectory = os.path.join(dir, "negatives")

    trainingDataLabels = []
    trainingData = []

    validationDataLabels = []
    validationData = []

    i = 1
    files = None
    if maxSize > 0:
        files = [f for f in os.listdir(positivesDirectory)[:maxSize] if f.endswith(".json")]
    else:
        files = [f for f in os.listdir(positivesDirectory) if f.endswith(".json")]


    for file in files:
        if file.endswith(".json"):
            if i % 100 == 0:
                print(str(i) + "/" + str(len(files)))

            if np.random.random_sample() <= percentageOfDataIsValidation:
                validationData.append(getObject(os.path.join(positivesDirectory, file)))
                validationDataLabels.append(np.array([1, 0]))
            else:
                trainingData.append(getObject(os.path.join(positivesDirectory, file)))
                trainingDataLabels.append(np.array([1, 0]))

            i = i + 1

    if maxSize > 0:
        files = [f for f in os.listdir(negativesDirectory)[:maxSize] if f.endswith(".json")]
    else:
        files = [f for f in os.listdir(negativesDirectory) if f.endswith(".json")]

    i = 1
    for file in files:
        if file.endswith(".json"):
            if i % 100 == 0:
                print(str(i) + "/" + str(len(files)))

            if np.random.random_sample() <= percentageOfDataIsValidation:
                validationData.append(getObject(os.path.join(negativesDirectory, file)))
                validationDataLabels.append(np.array([0, 1]))
            else:
                trainingData.append(getObject(os.path.join(negativesDirectory, file)))
                trainingDataLabels.append(np.array([0, 1]))

            i = i + 1

    dict = {}
    dict["TRAINING_DATA"] = np.array(trainingData)
    dict["TRAINING_LABELS"] = np.array(trainingDataLabels)

    dict["VALIDATION_DATA"] = np.array(validationData)
    dict["VALIDATION_LABELS"] = np.array(validationDataLabels)

    return dict

# create a weight that's a normal distribution about 0 w/ stdev of .1
def createWeight(shape, name):
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

def buildNetworkVariables(frameWidth, frameHeight, poolSize, outputNeuronCount):
    varDict = {}

    varDict["W_conv1"] = createWeight([5, 5, 1, 8], "Conv Layer 1")
    varDict["b_conv1"] = createBias([8])

    varDict["W_conv2"] = createWeight([5, 5, 8, 16], "Conv Layer 2")
    varDict["b_conv2"] = createBias([16])

    varDict["W_conv3"] = createWeight([2, 2, 16, 32], "Conv Layer 3")
    varDict["b_conv3"] = createBias([32])

    newWidth = int(math.ceil(frameWidth / (poolSize * poolSize / 2 * poolSize / 2)))
    newHeight = int(math.ceil(frameHeight / (poolSize * poolSize / 2 * poolSize / 2)))

    varDict["W_fc1"] = createWeight([newHeight * newWidth * 32, 1024], "Fully connected layer")
    varDict["b_fc1"] = createBias([1024])

    varDict["W_output"] = createWeight([1024, outputNeuronCount], "Output layer")
    varDict["b_output"] = createBias([outputNeuronCount])

    return varDict

#batchSize = 2
#frameWidth = 256
#frameHeight = 212
#frameSize = frameWidth * frameHeight
#outputNeuronCount = 2
#poolSize = 4
def buildNetwork(x, labels, probOfDropout, varDict, outputNeuronCount, frameWidth, frameHeight, poolSize, forTraining):

    reformatedImaged = tf.reshape(x, [-1, frameWidth, frameHeight, 1])

    conv1Handle = tf.nn.relu(conv2d(reformatedImaged, varDict["W_conv1"]) + varDict["b_conv1"])
    pool1Handle = max_pooling(conv1Handle, poolSize)

    conv2Handle = tf.nn.relu(conv2d(pool1Handle, varDict["W_conv2"]) + varDict["b_conv2"])
    pool2Handle = max_pooling(conv2Handle, poolSize / 2)

    conv3Handle = tf.nn.relu(conv2d(pool2Handle, varDict["W_conv3"]) + varDict["b_conv3"])
    pool3Handle = max_pooling(conv3Handle, poolSize / 2)

    newWidth = int(math.ceil(frameWidth / (poolSize * poolSize / 2 * poolSize / 2)))
    newHeight = int(math.ceil(frameHeight / (poolSize * poolSize / 2 * poolSize / 2)))

    pool3FlatLayer = tf.reshape(pool3Handle, [-1, newHeight * newWidth * 32])
    fullyConnectedLayerHandle = tf.nn.relu(tf.matmul(pool3FlatLayer, varDict["W_fc1"]) + varDict["b_fc1"])

    handleDropoutLayer = tf.nn.dropout(fullyConnectedLayerHandle, probOfDropout)

    handleOfOutputLayer = tf.matmul(handleDropoutLayer, varDict["W_output"]) + varDict["b_output"]

    if forTraining:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = handleOfOutputLayer))
        trainer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        return trainer
    else:
        correctArray = tf.equal(tf.argmax(handleOfOutputLayer, 1), tf.argmax(labels, 1))
        correctArrayAsInt = tf.to_float(correctArray)
        accuracy = tf.reduce_mean(correctArrayAsInt)
        return accuracy


#pickle.dump(trainingData, open("D:/cnnData/training_justjson/data.pickle",
#"wb"))

#print("Loading data...")
#trainingData = pickle.load(open("D:/cnnData/training_justjson/data.pickle",
#"rb"))
#print("...done")
with tf.Session() as session:
    trainingData = readTrainingData("D:/cnnData/training", -1, .1)

    poolSize = 4
    trainingBatchSize = 2
    outputNeuronCount = 2
    frameWidth = 256
    frameHeight = 212
    trainingBatchSize = 2

    # build the variables to be used across the training and testing graphs
    varDict = buildNetworkVariables(frameWidth, frameHeight, poolSize, outputNeuronCount)

    frameSize = frameWidth * frameHeight

    # create the training placeholders.
    trainingX = tf.placeholder(dtype = tf.float32, shape = [trainingBatchSize, frameSize])
    trainingLabels = tf.placeholder(dtype=tf.int32, shape= [trainingBatchSize, outputNeuronCount])
    trainingDropoutProb = tf.placeholder(dtype=tf.float32)

    # build the training graph
    trainer = buildNetwork(trainingX, trainingLabels, trainingDropoutProb, varDict, 
                           outputNeuronCount, frameWidth, frameHeight, poolSize, True)

    # bulid the saver.
    saver = tf.train.Saver()

    # create the testing placeholders.
    testingX = tf.placeholder(dtype = tf.float32, shape = [len(trainingData["VALIDATION_DATA"]), frameSize])
    testingLabels = tf.placeholder(dtype=tf.int32, shape= [len(trainingData["VALIDATION_DATA"]), outputNeuronCount])
    testingDropoutProb = tf.placeholder(dtype=tf.float32)

    # build the testing graph
    accuracy = buildNetwork(testingX, testingLabels, testingDropoutProb, varDict, 
                            outputNeuronCount, frameWidth, frameHeight, poolSize, False)

    session.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("C:/users/brush/desktop/deepLearning/logs", session.graph)
    writer.close()

    session.run(tf.initialize_all_variables())

    for i in range(0, 20000):
        batch = next_batch(trainingBatchSize, trainingData["TRAINING_DATA"], trainingData["TRAINING_LABELS"])
        trainer.run(feed_dict = {trainingX: batch[0], trainingLabels: batch[1], trainingDropoutProb: .5})

        if i % 25 == 0:

            averageAccuracy = accuracy.eval(feed_dict={testingX:trainingData["VALIDATION_DATA"], 
                                                       testingLabels:trainingData["VALIDATION_LABELS"], testingDropoutProb: 1.0})
            print()
            print('step %d, training accuracy %g' % (i, averageAccuracy))

            with open("c:/users/brush/desktop/deepLearning/results.csv", "a") as results:
                results.write(str(averageAccuracy) + "\n")
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

        saver.save(session, "C:/users/brush/desktop/deepLearning/checkPoint.ckpt")

