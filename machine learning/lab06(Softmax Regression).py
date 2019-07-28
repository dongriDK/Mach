import tensorflow as tf

# # 이론
# x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],
#         [1,2,5,6],[1,6,6,6],[1,7,7,7]]
# y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
#
# # tf.placeholder -> tf.compat.v1.placeholder
# X = tf.compat.v1.placeholder("float", [None, 4])
# Y = tf.compat.v1.placeholder("float", [None, 3])
# nb_classes = 3
#
# # tf.random_normal -> tf.random.normal
# W = tf.Variable(tf.random.normal([4, nb_classes]), name='weight')
# b = tf.Variable(tf.random.normal([nb_classes]), name='bias')
#
# # hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# # # tf.log -> tf.math.log
# # cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hypothesis), axis = 1))
#
# # Fancy
# logits = tf.matmul(X, W) + b
# hypothesis = tf.nn.softmax(logits)
# cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels==Y_one_hot)
# cost = tf.reduce_mean(cost_i)
#
#
# # tf.train.GradientDescentOptimizer -> tf.compat.v1.train.GradientDescentOptimizer
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# # tf.Session -> tf.compat.v1.Session
# with tf.compat.v1.Session() as sess:
#     # tf.global_variables_initializer -> tf.compat.v1.global_variables_initializer
#     sess.run(tf.compat.v1.global_variables_initializer())
#
#     for step in range(20001):
#         sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
#         if step % 200 == 0:
#             print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
#
#     all = sess.run(hypothesis, feed_dict={X:[[1,11,7,9], [1,3,4,3], [1,1,0,1]]})
#     # tf.arg_max -> tf.math.argmax
#     print(all, sess.run(tf.math.argmax(all, 1)))

# # 실습
import tensorflow as tf
import numpy as np

xy = np.loadtxt('zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

X = tf.compat.v1.placeholder(tf.float32, [None, 16])
Y = tf.compat.v1.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random.normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# softmax_cross_entropy_with_logits -> tf.
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(20001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        # if step % 200 == 0:
        #     loss, acc = sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})
            # print("step : {:5}\tLost: {:.3f}\tAcc:{:.2%}".format(step, loss, acc))

        pred = sess.run(prediction, feed_dict={X:x_data})
        for p, y in zip(pred, y_data.flatten()):
            print("[{}] Prediction: {} True Y: {}".format(p==int(y), p, int(y)))
