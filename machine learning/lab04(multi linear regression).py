

# file에서 data 읽어오기
import tensorflow as tf
filename_queue = tf.train.string_input_producer(
        ['score_test.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 각 각 필드의 데이터 타입 정의 (float32)
record_defaults = [[0.], [0.], [0.], [0.]]
# value를 csv decode 해라
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch(
        [xy[0:-1], xy[-1:]], batch_size=10)

# X value의 개수 3개
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={X:x_batch, Y:y_batch})
    if step % 200 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)

coord.request_stop()
coord.join(threads)


# # multi_variable linear regression
# x_data = [[73., 80., 75.], [93., 88., 93.],
#         [89., 91., 90.], [96., 98., 100.,], [73., 66., 70.]]
#
# y_data = [[152.], [185.], [180.], [196.], [142.]]
#
# # 데이터 개수는 None 한 데이터의 열 수는 3
# X = tf.placeholder(tf.float32, shape=[None, 3])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
#
# # X로부터 입력 데이터 개수 3, 출력 되는 Y개수 1개
# W = tf.Variable(tf.random_normal([3, 1]), name='weight')
# # 출력되는 Y의 개수 1개
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# hypothesis = tf.matmul(X, W) + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(2001):
#     cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
#     if step % 200 == 0:
#         print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)
