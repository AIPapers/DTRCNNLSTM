import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.utils import shuffle

import preprocessData as preprocess

image_size = 64
num_labels = 10
num_channels = 1
finalImageSize = 4

print("Setting up CNN...")


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def weight_variable(shape, name=""):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)


def bias_variable(shape, name=""):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x, name=""):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)


x = tf.placeholder(tf.float32, [None, 64, 64, 1])
W_conv1 = weight_variable([5, 5, 1, 64], name='W1')
b_conv1 = bias_variable([64], name='B1')

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1, name='Conv1')
h_pool1 = max_pool_2x2(h_conv1, name='Pool1')

W_conv2 = weight_variable([5, 5, 64, 256], name='W2')
b_conv2 = bias_variable([256], name='B2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='Conv2')
h_pool2 = max_pool_2x2(h_conv2, name='Pool2')

# print(h_conv1.get_shape())
# print(h_conv2.get_shape())
# print(h_pool1.get_shape())
# print(h_pool2.get_shape())

W_fc1 = weight_variable([4 * 4 * 64, 1024], name='W3')
b_fc1 = bias_variable([1024], name='B3')

lstm_pool = tf.reshape(h_pool2, [-1, 4, 4 * 256], name="PoolLSTM")
lstm1 = tf.nn.rnn_cell.LSTMCell(256, state_is_tuple=True)
val, state = tf.nn.dynamic_rnn(lstm1, lstm_pool, dtype=tf.float32)

# print(val.get_shape(), "Val Shape")

keep_prob = tf.placeholder("float", name='KeepProb')
lstm_drop = tf.nn.dropout(val, keep_prob, name="DropLSTM")

# print(lstm_drop.get_shape(), "LSTM_DROP")

h_pool2_flat = tf.reshape(val, [-1, 1024], name='Pool3')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, 'MatMult3')

keep_prob = tf.placeholder("float", name='KeepProb')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='Drop4')

W_fc2 = weight_variable([1024, 10], name='W5')
b_fc2 = bias_variable([10], name='B5')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_ = tf.placeholder(tf.float32, [None, 10], name='Ytruth')

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)), name='CrossEntropy'))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
loss = tf.reduce_mean(cross_entropy)
tf.summary.scalar('loss', loss)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='CorrectPrediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='Accuracy')

saver = tf.train.Saver()

sess = tf.InteractiveSession()

print("Loading Data")
lb = preprocessing.LabelBinarizer()
xTrain, yTrain, xTest, yTest = preprocess.preProcessData()
yTrain = lb.fit_transform(yTrain)
yTest = lb.fit_transform(yTest)
print(xTrain.shape, np.expand_dims(xTrain[0], axis=0).shape, yTrain.shape)

N = xTrain.shape[0]

do_training = True

if do_training:
    sess.run(tf.global_variables_initializer())

    totalAccuracy = []
    totalTrainAccuracy = []
    lossTotal = []

    num_epochs = 20
    for j in range(num_epochs):

        print("Shuffling training data")
        xTrain, yTrain = shuffle(xTrain, yTrain, random_state=0)
        accuracyBatch = []
        lossBatch = []

        for i in range(0, N - 5 * 17, 5 * 17):
            if i % (20 * 17) == 0:
                print("Training %d step" % i)
            train, lossVal, accuracyTrain = sess.run([train_step, loss, accuracy],
                                                     feed_dict={x: xTrain[i:i + 5 * 17], y_: yTrain[i:i + 5 * 17],
                                                                keep_prob: 0.5})
            lossBatch.append(lossVal)
            accuracyBatch.append(100 * accuracyTrain)

        lossTotal.append(np.average(lossBatch))
        totalTrainAccuracy.append(np.average(accuracyBatch))
        print("Finished training - %d, with accuracy %g." % ((j + 1), np.average(accuracyBatch)))
        ti = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('time:%s\n' % ti)
        print("Average Loss for epoch - %d - %g" % ((j + 1), np.average(lossBatch)))

        acc = []
        for i in range(0, xTest.shape[0], 1000):
            acc.append(100 * accuracy.eval(feed_dict={x: xTest[i:i + 1000], y_: yTest[i:i + 1000], keep_prob: 1.0}))

        print("Epoch - %d, Accuracy: %g\n" % (j + 1, np.average(acc)))
        totalAccuracy.append(np.average(acc))
        del acc

    del xTrain, yTrain, xTest, yTest
    save_path = saver.save(sess, "saved_models/model_%d_lstm.ckpt" % N)
    print("Model saved in file: ", save_path)

    xData = [i for i in range(1, num_epochs + 1)]
    xDataLoss = [i for i in range(1, len(lossTotal) + 1)]

    print(totalAccuracy)
    print(totalTrainAccuracy)
    print(lossTotal)
    plt.plot(xData, totalTrainAccuracy)
    plt.plot(xData, totalAccuracy)
    plt.savefig("saved_models/model_%d_lstm_accuracyTrainTestPlot.png" % N)

    plt.savefig("saved_models/model_%d_lstm_accuracyPlot.png" % N)

    lossPlot = plt.figure(2)
    plt.plot(xDataLoss, lossTotal)
    plt.savefig("saved_models/model_%d_lstm_lossPlot.png" % N)

    plt.show()
    print("Minimized loss = %g" % (lossTotal[len(lossTotal) - 1]))

else:

    model_name = "saved_models/model_%d_lstm.ckpt" % N
    print("Loading model '%s'" % model_name)
    saver.restore(sess, model_name)
    print("Model restored.")

    acc = []
    for i in range(0, xTest.shape[0], 1000):
        acc.append(100 * accuracy.eval(feed_dict={x: xTest[i:i + 1000], y_: yTest[i:i + 1000], keep_prob: 1.0}))

    print("Accuracy: %g\n" % (np.average(acc)))
