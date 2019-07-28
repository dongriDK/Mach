import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]
# W = tf.Variable(tf.random_normal([1]), name="weight")
W = tf.Variable(5.0)
# W = tf.placeholder(tf.float32)
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 자동
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
# 수동
# learning_rate = 0.1
# gradient = tf.reduce_mean((W * X - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 자동
for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
# 수동
# for step in range(21):
#     sess.run(update, feed_dict={X:x_data, Y:y_data})
#     print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

# 그래프 그리기
W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W:feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()
