import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

IMG_SIZE_PX = 50
SLICE_COUNT = 20
KEEP_RATE = 0.8
n_classes = 2
batch_size = 10


training_data = input("Please enter the training data file with directory: ")
test_data = input("Please enter the test data file with directory: ")
numModel = input("Please enter the number of models you wish to train: ")
print("Creating folder to hold model")
try:
    os.mkdir(os.getcwd() + "/" + "modeldirect")
except OSError:
    print("This path already exists overwriting the existing model")
saveM = [os.getcwd() + "/modeldirect/model" + str(i) for i in range(numModel)]


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x, keep_rate=KEEP_RATE):

    weights = {'W_conv1': tf.get_variable('W_conv1', shape=[3, 3, 3, 1, 32], initializer=tf.contrib.layers.variance_scaling_initializer()),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2': tf.get_variable('W_conv2', shape=[3, 3, 3, 32, 64], initializer=tf.contrib.layers.variance_scaling_initializer()),

               'W_fc': tf.get_variable('W_fc', shape=[54080, 1024], initializer=tf.contrib.layers.xavier_initializer()),
               'out': tf.get_variable('out', shape=[1024, n_classes], initializer=tf.contrib.layers.xavier_initializer())}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def shuffle_data(training_data):
    origional = np.load(training_data)
    # numbers = [i for i in range(len(origional))]
    numbers = random.sample([i for i in range(len(origional))], len(origional))
    shuffle = [origional[i] for i in numbers]
    return shuffle


def train_neural_network(x, k):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    saver = tf.train.Saver()

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in tqdm(train_data):
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    pass

            print('Epoch', epoch + 1, 'completed out of',
                  hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval(
                {x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

            saver.save(sess, saveM[k])

        print('Done. Finishing accuracy:')
        print('Accuracy:', accuracy.eval(
            {x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

        print('fitment percent:', successful_runs / total_runs)
        saver = tf.train.Saver()
        saver.save(sess, saveM[k])


def submit(x, k):

    prediction = convolutional_neural_network(x, keep_rate=1.)
    probabilities = tf.nn.softmax(prediction)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # tf.reset_default_graph()
        saver.restore(sess, saveM[k])
        print("Model-" + str(k) + " restored!")
        eval_data = np.load(test_data)

        sol = []
        for data in eval_data:
            X = data[0]
            id = data[1]
            probs = probabilities.eval(feed_dict={x: X})
            pred = prediction.eval(feed_dict={x: X})

            sol.append([id, probs[0, 1]])

        np.save(os.getcwd() + "/" + "modeldirect/model-" +
                str(k) + "-predictions" + ".npy", sol)


def modelaccuracy(test_data, numModel):
    Allmodelprob = [0 for i in range(len(np.load(test_data)))]
    for i in range(numModel):
        t = np.load(os.getcwd() + "/modeldirect/model-" +
                    str(i) + "-predictions" + ".npy")
        acc = 0
        for j in range(len(t)):
            if t[j][1] >= 0.5:
                label = np.array([0, 1])
            else:
                label = np.array([1, 0])
                Allmodelprob[j] += 1
            if label[0] == t[j][0][0]:
                acc += 1
        print("Model " + str(i + 1) + " accuracy = " + str(acc) +
              "/" + str(len(t)) + " = " + str(acc / len(t)))

    Allmodelprob = [i / numModel for i in Allmodelprob]
    finalacc = 0
    for k in range(len(t)):
        if Allmodelprob[k] <= 0.5:
            label = np.array([0, 1])
        else:
            label = np.array([1, 0])
        if label[0] == t[k][0][0]:
            finalacc += 1
    print("Final accuracy of the combined models: " + str(finalacc / len(t)))
    for i in range(len(t)):
        print(t[i][0], Allmodelprob[i])


if __name__ == "__main__":

    much_data = shuffle_data(training_data)

    for k in range(numModel):
        tf.reset_default_graph()
        x = tf.placeholder('float')
        y = tf.placeholder('float')

        train_data = much_data[:-25]
        validation_data = much_data[-25:]

        train_neural_network(x, k)

    for k in range(numModel):
        tf.reset_default_graph()
        x = tf.placeholder('float')
        y = tf.placeholder('float')
        submit(x, k)

    modelaccuracy(test_data, numModel)
