import tensorflow as tf
"""

filename_queue = tf.train.string_input_producer(
        ['winequality-red.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 각 각 필드의 데이터 타입 정의 (float32)
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
# value를 csv decode 해라
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch(
        [xy[0:-1], xy[-1:]], batch_size=1)

# X value의 개수 3개
X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([11, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(20001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                feed_dict={X:x_batch, Y:y_batch})
        if step % 1000 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)

    print(sess.run(hypothesis, feed_dict={X:[[6, 0.50, 0.88, 3.1, 0.051, 20, 88.0, 0.992, 3.50, 0.87, 12.9]]}))

    coord.request_stop()
    coord.join(threads)


import tensorflow as tf
"""


### tensorboard
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
# batch_xs, batch_ys = mnist.train.next_batch(100)
nb_classes = 10

X = tf.compat.v1.placeholder(tf.float32, [None, 784])
Y = tf.compat.v1.placeholder(tf.float32, [None, nb_classes])
with tf.name_scope("Layer1"):
    W = tf.Variable(tf.random.normal([784, nb_classes]))
    b = tf.Variable(tf.random.normal([nb_classes]))
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

    tf.compat.v1.summary.histogram("W1", W)
    tf.compat.v1.summary.histogram("b", b)
    tf.compat.v1.summary.histogram("Hypothesis", hypothesis)

with tf.name_scope("Cost"):
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hypothesis), axis=1))
    tf.compat.v1.summary.scalar("Cost", cost)

with tf.name_scope("Train"):
    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
tf.compat.v1.summary.scalar("accuracy", accuracy)
training_epochs = 15
batch_size = 100

with tf.compat.v1.Session() as sess:
    merged_summary = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter("./logs/mnist_logs_01")
    writer.add_graph(sess.graph)

    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _, summary, acc = sess.run([cost, train, merged_summary, accuracy], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c / total_batch
            writer.add_summary(summary, global_step=epoch)

        print ('Epoch:', '%02d' % (epoch + 1),'Accuracy : ', acc,'cost =', '{:.9f}'.format(avg_cost))

    print ("Accuracy : ", accuracy.eval(session = sess, feed_dict={X:mnist.test.images, Y: mnist.test.labels}))

    # r = random.randint(0, mnist.test.num_examples -1)
    # print("Label:", sess.run(tf.math.argmax(mnist.test.labels[r:r+1], 1)))
    # print("Prediction:", sess.run(tf.math.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}))
    # plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
    # plt.show()

"""
# winequality - failed
# filename_queue = tf.train.string_input_producer(
#         ['winequality-red.csv'], shuffle=False, name='filename_queue')
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
#
# # 각 각 필드의 데이터 타입 정의 (float32)
# record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
# # value를 csv decode 해라
# xy = tf.decode_csv(value, record_defaults=record_defaults)
#
# train_x_batch, train_y_batch = tf.train.batch(
#         [xy[0:-1], xy[-1:]], batch_size=1)
#
# # X value의 개수 3개
# X = tf.compat.v1.placeholder(tf.float32, shape=[None, 11])
# Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
#
# with tf.name_scope("Layer1"):
#     W = tf.Variable(tf.random.normal([11, 1]), name='weight')
#     b = tf.Variable(tf.random.normal([1]), name='bias')
#
#     hypothesis = tf.matmul(X, W) + b
#     tf.compat.v1.summary.histogram("Hypothesis", hypothesis)
#
# with tf.name_scope("Cost"):
#     cost = tf.reduce_mean(tf.square(hypothesis - Y))
#     tf.compat.v1.summary.scalar("Cost", cost)
#
# with tf.name_scope("Train"):
#     optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5)
#     train = optimizer.minimize(cost)
#
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# tf.compat.v1.summary.scalar("accuracy",accuracy)
#
# with tf.compat.v1.Session() as sess:
#     merged_summary = tf.compat.v1.summary.merge_all()
#     writer = tf.compat.v1.summary.FileWriter("./logs/realworld_logs_01")
#     writer.add_graph(sess.graph)
#     sess.run(tf.compat.v1.global_variables_initializer())
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     for step in range(20001):
#         x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
#         _, summary, cost_val = sess.run([train, merged_summary, cost], feed_dict={X:x_batch, Y:y_batch})
#         writer.add_summary(summary, global_step=step)
#         # cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
#         #         feed_dict={X:x_batch, Y:y_batch})
#         if step % 1000 == 0:
#             print(step, "Cost: ", cost_val, "\nPrediction: \n", summary)
#
#     # print(sess.run(hypothesis, feed_dict={X:[[6, 0.50, 0.88, 3.1, 0.051, 20, 88.0, 0.992, 3.50, 0.87, 12.9]]}))
#
#     coord.request_stop()
#     coord.join(threads)
"""
