
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
with tf.compat.v1.Session() as sess:
    for i in range(5):
        r = random.randint(0, mnist.test.num_examples -1)

        # print("Label:", sess.run(tf.math.argmax(mnist.test.labels[r:r+1], 1)))
        print("Label:", sess.run(tf.math.argmax(mnist.test.labels[r:r+1], 1)))
        # print("Prediction:", sess.run(tf.math.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}))
        plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
        plt.show()
