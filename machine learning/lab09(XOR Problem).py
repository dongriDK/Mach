import tensorflow as tf
import numpy as np

# # Neural Net for XOR
# x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
# y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
#
# X = tf.compat.v1.placeholder(tf.float32)
# Y = tf.compat.v1.placeholder(tf.float32)
# W1 = tf.Variable(tf.random.normal([2, 10]), name =  'weight')
# b1 = tf.Variable(tf.random.normal([10]), name='bias')
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
#
# W2 = tf.Variable(tf.random.normal([10, 1]), name =  'weight')
# b2 = tf.Variable(tf.random.normal([1]), name='bias')
# hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
# # hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#
# cost = -tf.reduce_mean(Y * tf.math.log(hypothesis) + (1-Y)*tf.math.log(1-hypothesis))
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#
#     for step in range(10001):
#         sess.run(train, feed_dict={X:x_data, Y:y_data})
#         if step % 100 == 0:
#             print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run([W1,W2]))
#
#     h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
#     # print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run(W2))
#     print("\n hypothesis : ", h, "\nCorrect : ", c, "\nAccuracy : ", a)


# Neural Net for XOR using tensorBoard
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.compat.v1.placeholder(tf.float32, [None, 2], name='x')
Y = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y')

with tf.name_scope("Layer1"):
    W1 = tf.Variable(tf.random.normal([2, 10]), name =  'weight_1')
    b1 = tf.Variable(tf.random.normal([10]), name='bias_1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    tf.compat.v1.summary.histogram("W1", W1)
    tf.compat.v1.summary.histogram("b1", b1)
    tf.compat.v1.summary.histogram("Layer1", layer1)

with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random.normal([10, 1]), name =  'weight_2')
    b2 = tf.Variable(tf.random.normal([1]), name='bias_2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    tf.compat.v1.summary.histogram("W2", W2)
    tf.compat.v1.summary.histogram("b2", b2)
    tf.compat.v1.summary.histogram("Hypothesis", hypothesis)

with tf.name_scope("Cost"):
    cost = -tf.reduce_mean(Y * tf.math.log(hypothesis) + (1-Y)*tf.math.log(1-hypothesis))
    tf.compat.v1.summary.scalar("Cost", cost)

with tf.name_scope("Train"):
    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
tf.compat.v1.summary.scalar("acuuracy", accuracy)

with tf.compat.v1.Session() as sess:
    merged_summary = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter("./logs/xor_logs_01")
    writer.add_graph(sess.graph)
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(10001):
        _, summary, cost_val = sess.run([train, merged_summary, cost], feed_dict={X:x_data, Y:y_data})
        writer.add_summary(summary, global_step=step)
        # sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 1000 == 0:
            print(step, cost_val)
            # print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run([W1,W2]))

    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    # print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run(W2))
    print("\n hypothesis : ", h, "\nCorrect : ", p, "\nAccuracy : ", a)

# MNist wide deep Neural Net learning and tensorBoard

# from tensorflow.examples.tutorials.mnist import input_data
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import random
# mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
# # batch_xs, batch_ys = mnist.train.next_batch(100)
# nb_classes = 400
#
# X = tf.compat.v1.placeholder(tf.float32, [None, 784])
# Y = tf.compat.v1.placeholder(tf.float32, [None, 10])
# W1 = tf.Variable(tf.random.normal([784, nb_classes]))
# b1 = tf.Variable(tf.random.normal([nb_classes]))
# layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)
#
# W2 = tf.Variable(tf.random.normal([nb_classes, 10]))
# b2 = tf.Variable(tf.random.normal([10]))
# hypothesis = tf.nn.softmax(tf.matmul(layer1, W2) + b2)
#
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hypothesis), axis=1))
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#
# is_correct = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#
# training_epochs = 20
# batch_size = 100
#
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     for epoch in range(training_epochs):
#         avg_cost = 0
#         total_batch = int(mnist.train.num_examples / batch_size)
#
#         for i in range(total_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
#             avg_cost += c / total_batch
#         print ('Epoch:', '%02d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
#
#     print ("Accuracy : ", accuracy.eval(session = sess, feed_dict={X:mnist.test.images, Y: mnist.test.labels}))
#
#     r = random.randint(0, mnist.test.num_examples -1)
#     print("Label:", sess.run(tf.math.argmax(mnist.test.labels[r:r+1], 1)))
#     print("Prediction:", sess.run(tf.math.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}))
#     plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
#     plt.show()
#
# # tensorBoard
# s2_hist = tf.summary.histogram("weights2", W2)
# cost_summ = tf.summary.scalar("cost", cost)
#
# summary = tf.summary.merge_all()
#
# writer = tf.summary.FileWriter('./logs')
# writer.add_graph.eval(sess=sess, sess.graph)
# s, _ = sess.run([summary, optimkizer], feed_dict=fedd_dict)
# writer.add_summary(s, global_step=global_step)
