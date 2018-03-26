import tensorflow as tf
import numpy as np
from PIL import Image


def process_images(img):
    img = img.resize((500, 500), Image.ANTIALIAS)
    r, g, b = img.split()
    r_arr = np.array(r)
    g_arr = np.array(g)
    b_arr = np.array(b)

    r_arr = r_arr.astype(np.float64)
    g_arr = g_arr.astype(np.float64)
    b_arr = b_arr.astype(np.float64)

    r_arr -= np.mean(r_arr, axis=0)
    g_arr -= np.mean(g_arr, axis=0)
    b_arr -= np.mean(b_arr, axis=0)

    r1 = Image.fromarray(r_arr).convert("L")
    g1 = Image.fromarray(g_arr).convert("L")
    b1 = Image.fromarray(b_arr).convert("L")
    img1 = Image.merge("RGB", (r1, g1, b1))
    img1 = img1.resize((100, 100), Image.ANTIALIAS)
    img1 = img1.convert("L")
    return img1


img = Image.open("3.jpg")
img = process_images(img)

lim = np.asarray(img, dtype='float64')
lim /= 256
lim = lim.reshape(1, 100, 100, 1)

X = tf.placeholder("float", [None, 100, 100, 1])


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
predict_op = tf.argmax(py_x, axis=1)



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, "d:\\model\\model-epoch-199.ckpt")
    test_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
    test_indices = np.asarray(test_indices, dtype='int64')
    print(sess.run(predict_op, feed_dict={X: lim, p_keep_conv: 1.0, p_keep_hidden: 1.0})[0])
