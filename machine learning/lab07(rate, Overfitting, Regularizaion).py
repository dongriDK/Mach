import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
# overshooting : learning_rate을 크게주면 그래프 밖으로 튕겨저나감 숫자말고 이상한 값 출력됨
# 처음을 0.01로 잡고 발산하면 줄이고 중간에 멈추면 크게
# cost값이 발산하거나 이상한 값을 보이면 데이터중에 차이가 큰게 있는지 확인, preprocessing했는지 확인
# overfitting : linear하지 않고 구부러진 regression
# online learning : 이미 학습한 데이터를 가지고 새로운 데이터가 들어왔을 때 이어서 학습하는 방법

# Training and test sets
# x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4],  [1, 5, 5], [1, 7, 5],
#           [1, 2, 5], [1, 6, 6], [1, 7, 7]]
# y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0],
#           [0, 1, 0], [1, 0, 0], [1, 0, 0]]
#
# x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
# y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
#
# X = tf.compat.v1.placeholder("float", [None, 3])
# Y = tf.compat.v1.placeholder("float", [None, 3])
# W = tf.Variable(tf.random.normal([3, 3]))
# b = tf.Variable(tf.random.normal([3]))
#
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hypothesis), axis=1))
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#
# prediction = tf.arg_max(hypothesis, 1)
# is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     for step in range(201):
#         cost_val, w_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
#         print(step, cost_val, w_val)
#
#     print("Prediction: ", sess.run(prediction, feed_dict={X: x_test}))
#     print("Accuracy: ", sess.run(accuracy, feed_dict={X:x_test, Y:y_test}))

# realworld test (MNIST Dataset)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
# batch_xs, batch_ys = mnist.train.next_batch(100)
nb_classes = 10

X = tf.compat.v1.placeholder(tf.float32, [None, 784])
Y = tf.compat.v1.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random.normal([784, nb_classes]))
b = tf.Variable(tf.random.normal([nb_classes]))
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c / total_batch
        print ('Epoch:', '%02d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print ("Accuracy : ", accuracy.eval(session = sess, feed_dict={X:mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples -1)
    print("Label:", sess.run(tf.math.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.math.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
    plt.show()

# Normalized input (MinMaxScaler()) - failed
# normalized는 됐지만 원하는 데이터 결과도 normalized되어서 나옴
# x = [2104, 1600, 2400, 1416, 3000, 1985, 1534]
# y = [400, 330, 369, 232, 540, 300, 315]
# xy = np.array([[2104,400],[1600,330],[2400,369],[1416,232],[3000,540],[1985,300],[1534,315]])
# minmaxscaler = MinMaxScaler()
# xy = minmaxscaler.fit_transform(xy)
# print(xy)
# x = xy[:, 0:-1]
# y = xy[:, [-1]]
# # x = [[2104], [1600], [2400], [1416], [3000], [1985], [1534]]
# # y = [[400], [330], [369], [232], [540], [300], [315]]
# X = tf.placeholder(tf.float32, shape=[None, 1])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
#
# W = tf.Variable(tf.random.normal([1, 1]), name="weight")
# b = tf.Variable(tf.random.normal([1]), name="bias")
#
#
# hypothesis = X * W + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(2001):
#         _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={X:x, Y:y})
#         # if step % 20 == 0:
#         print(step, cost_val, W_val, b_val)
#
#     # print(sess.run(hypothesis, feed_dict={X: [[1427],[1380],[1494]]}))
#     # print(sess.run(hypothesis, feed_dict={X: [2.5]}))
#     # print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
