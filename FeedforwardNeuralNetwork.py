## Reid Bachman and Matt Fein

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 30
tf.set_random_seed(RANDOM_SEED)
stddev = 0.1
testSize = .33
randomState = RANDOM_SEED


def main():
    trainx, testx, trainy, testy = get_data()

    xsize = trainx.shape[1]
    hsize = 256
    ysize = trainy.shape[1]

    x = tf.placeholder("float", shape=[None, xsize])
    y = tf.placeholder("float", shape=[None, ysize])

    w_1 = initializeweights((xsize, hsize))
    w_2 = initializeweights((hsize, ysize))

    yprocessed = forwardprop(x, w_1, w_2)
    predict = tf.argmax(yprocessed, axis=1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yprocessed))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        for i in range(len(trainx)):
            sess.run(updates, feed_dict={x: trainx[i: i + 1], y: trainy[i: i + 1]})

        train_accuracy = np.mean(np.argmax(trainy, axis=1) ==
                                 sess.run(predict, feed_dict={x: trainx, y: trainy}))
        test_accuracy = np.mean(np.argmax(testy, axis=1) ==
                                sess.run(predict, feed_dict={x: testx, y: testy}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()


def initializeweights(shape):
    weights = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(weights)


def forwardprop(X, w1, w2):
    h = tf.nn.sigmoid(tf.matmul(X, w1))
    yProcessed = tf.matmul(h, w2)
    return yProcessed


def get_data():
    iris = datasets.load_iris()
    data = iris["data"]
    target = iris["target"]

    n, m = data.shape
    allx = np.ones((n, m + 1))
    allx[:, 1:] = data

    num_labels = len(np.unique(target))
    ally = np.eye(num_labels)[target]
    return train_test_split(allx, ally, test_size=testSize, random_state=RANDOM_SEED)


if __name__ == '__main__':
    main()
