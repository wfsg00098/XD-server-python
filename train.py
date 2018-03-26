import os
import pickle

import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


trX = pickle.load(open("data\\trX.pkl", 'rb'))
teX = pickle.load(open("data\\teX.pkl", 'rb'))
trY = pickle.load(open("data\\trY.pkl", 'rb'))
teY = pickle.load(open("data\\teY.pkl", 'rb'))

trX = trX.reshape(-1, 100, 100, 1)
teX = teX.reshape(-1, 100, 100, 1)

print("data loaded")

X = tf.placeholder("float", [None, 100, 100, 1])
Y = tf.placeholder("float", [None, 3])


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([3, 3, 128, 256])
w5 = init_weights([256 * 7 * 7, 625])
w_o = init_weights([625, 3])


def model(X, w, w2, w3, w4, w5, w_o, p_keep_connv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_connv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_connv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_connv)

    l4a = tf.nn.relu(tf.nn.conv2d(l3, w4, strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l4 = tf.reshape(l4, [-1, w5.get_shape().as_list()[0]])
    l4 = tf.nn.dropout(l4, p_keep_connv)

    l5 = tf.nn.relu(tf.matmul(l4, w5))
    l5 = tf.nn.dropout(l5, p_keep_hidden)

    pyx = tf.matmul(l5, w_o)
    return pyx


p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w5, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, axis=1)

batch_size = 128

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1000):
        print("epoch - " + str(i))
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        test_indices = np.asarray(test_indices, dtype='int64')

        print(np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                                                              p_keep_conv: 1.0,
                                                                                              p_keep_hidden: 1.0})))

        saver = tf.train.Saver()
        saver.save(sess, "d:\\model\\model-epoch-" + str(i) + ".ckpt")
        print("Saver end")
