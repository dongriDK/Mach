# Import data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
k = tf.matmul(x, W) + b
y = tf.nn.softmax(k)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
learning_rate = 0.5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(k, y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

print ("Training")
sess = tf.Session()
init = tf.global_variables_initializer() #.run()
sess.run(init)
for _ in range(1000):
    # 1000번씩, 전체 데이타에서 100개씩 뽑아서 트레이닝을 함.
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print ('b is ',sess.run(b))
print('W is',sess.run(W))
