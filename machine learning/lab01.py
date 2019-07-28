import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!");

sess = tf.Session() #Session을 만들고

# print(sess.run(hello)) #run으로 결과값 본다.

node1 = tf.constant(3.0, tf.float32) #constant : 정적 값
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# print(sess.run([node1, node2]))
# print(sess.run(node3))

a = tf.placeholder(tf.float32) #placeholder : 동적
b = tf.placeholder(tf.float32) #node 정의
adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))

# Ranks : 몇 차원이냐
# 0 : Scalar -> s = 123
# 1 : Vector -> v = [1.1, 2.2, 3.3]
# 2 : Matrix -> m = [[1,2,3], [4,5,6], [7,8,9]]
# 3 : 3-Tensor -> t = [[[2], [4], [6]], [[8], [10], [12]]]

# Shape : 각각 element에 몇 개 의 데이터가 들어있는가
# 0 : [] -> 0-D
# 1 : [D0] -> 1-D
# 2 : [D0, D1] -> 2-D
# 3 : [D0, D1, D2] -> 3-D

# => t = [[1,2,3], [4,5,6], [7,8,9]]
# [3 3) or [3, 3] 으로 표현

# Type
