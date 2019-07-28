import tensorflow as tf
# tf.compat.v1.set_random_seed(777)
# constant
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# Variable : tensorflow가 사용하는 Variable정의
# rank가 1인 tensor
W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# placeholder
X = tf.compat.v1.placeholder(tf.float32, shape=[None])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None])


hypothesis = X * W + b

# reduce_mean : 평균
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# magic
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# train : 그래프 모양이 됨 (minimize 됨)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()

# variable initializer
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, W, b, train],
        feed_dict={X:[1, 2, 3, 4, 5], Y:[2.1, 3.1, 4.1, 5.1, 6.1]})
    # sess.run(train)
    if step % 20 == 0:
        print(step, cost_val, w_val, b_val)


print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
