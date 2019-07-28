# reduce_mean : 평균 구하는 것 tf.reduce_mean([1,2]) -> 1 int형으로 나옴
# reduce_mean axis=1 : 행 평균, axis=0 : 열 평균
# reduce_sum, tf.math.argmax : 똑같음
# one_hot : tf.one_hot([[0], [1], [2]], depth=3).eval()
# -> array([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]], dtype=float32)
# one_hot 하면 한차원 더 많아짐 => tfl.reshape (t.shape([-1, 3]).eval()
# tf.cast : 형 바꾸기 tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
# -> array([1, 2, 3, 4], dtype=int32) / boolean형도 됨
# tf.stack : 여러 리스트들 쌓기
# tf.ones_like(list) : list와 같은 형태를 1로 채운다 / tf.zeros_like(list)
