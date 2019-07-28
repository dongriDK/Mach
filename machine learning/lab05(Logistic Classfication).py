import tensorflow as tf
import numpy as np

# data가  8개 제한
# x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
# y_daya = [[0], [0], [0], [1], [1], [1]]
#
# X = tf.placeholder(tf.float32, shape=[None, 2])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
# W = tf.Variable(tf.random_normal([2, 1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# # tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
#
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(10001):
#         cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y:y_daya})
#         if step % 200 == 0:
#             print(step, cost_val)
#
#             h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y:y_daya})
#             print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

# csv파일에서 가져옴
# numpy 사용
# xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]
#
# X = tf.placeholder(tf.float32, shape=[None, 8])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
#
# W = tf.Variable(tf.random_normal([8,1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     feed = {X:x_data, Y:y_data}
#     for step in range(10001):
#         sess.run(train, feed_dict=feed)
#         if step % 200 == 0:
#             print(step, sess.run(cost, feed_dict=feed))
#
#     h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
#     print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

# csv파일에서 가져옴
# tf.decode_csv 사용
filename_queue = tf.train.string_input_producer(['diabetes.csv'], shuffle=False, name='filename_queue')
# filename_queue = tf.data.Dataset.from_tensor_slices('diabetes.csv').shuffle(False)
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=1)

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])
InputX = tf.placeholder(tf.float32, shape=None)

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(10001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        # print(x_batch)
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_batch, Y:y_batch})
        if step % 1000 == 0:
            print("Cost : ", cost_val)
            # print(x_batch)
        #     print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)

    print(sess.run(hypothesis, feed_dict={X:[[0, 0, 0, 0, 0, 0, 0, 0]]}))
    coord.request_stop()
    coord.join(threads)
