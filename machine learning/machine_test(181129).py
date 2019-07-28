import numpy
from scipy import sparse
from IPython.display import display
import mglearn
###
# 1. NumPy 배열의 예
# x = numpy.array([[1, 2, 3], [4, 5, 6]])
# print ("x:\n{}".format(x))
###

###
# 2. 대각선 원소는 1이고 나머지는 0인 2차원 NumPy 배열을 만듬
# eye(숫자)는 몇 배열 만들지
# eye = numpy.eye(4)
# print("NumPy 배열 : \n{}".format(eye))
###

###
# 3. NumPy 배열을 CSR 포맷의  SciPy 희소 행렬로 변환
# - 0이 아닌 원소만 저장
# - 희소행렬은 값으로 의미가 있는 것들만 기록 함
# sparse_matrix = sparse.csr_matrix(eye)
# print ("SciPy의 CSR 행렬:\n{}".format(sparse_matrix))
###

###
# 4. COO 포맷을 이용한 희소 행렬 만들기
# data = numpy.ones(4)
# row_indices = numpy.arange(4)
# col_indices = numpy.arange(4)
# eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
# print("COO 표현:\n{}".format(eye_coo))
###

###
# 5. matplotlib으로 sin함수 그리기 (이미지 안나옴)
import matplotlib.pyplot as plt
# x = numpy.linspace(-10, 10, 100)
# y = numpy.sin(x)
# plt.plot(x, y, marker = "x")
###

###
# 6. pandas 모듈 사용
# - 데이터 처리와 분석 위한 파이썬 라이브러리
# - NumPy와는 달리 각 열의 타입이 달라도 됨
# - SQL, Excell, CSV 파일 같은 다양한 파일과 DB에서 데이터 읽을 수 있음
# - 딕셔너리 사용해서 DataFrame만드는 예제
import pandas
# data = {'Name': ["John", "Anna", "Peter", "Linda"],
#         'Location' : ["New York", "Paris", "Berlin", "London"],
#         'Age' : [24, 13, 53, 33]
#         }
# data_pandas = pandas.DataFrame(data)
# display(data_pandas) # 주피터 노트북에서 사용 가능
# print(data_pandas)
###

###
# 7. 데이터 적재
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# iris_dataset = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(
#     iris_dataset['data'], iris_dataset['target'], random_state=0)
# print ("iris_dataset의 키: \n{}".format(iris_dataset.keys()))
# print("타깃의 이름: {}".format(iris_dataset['target_names']))
# print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))
###

###
# iris_dataframe = pandas.DataFrame(X_train, columns=iris_dataset.feature_names)
# pandas.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
#     hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
###

###
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)
# X_new = numpy.array([[5, 2.9, 1, 0.2]])
# print("X_new.shape:{}".format(X_new.shape))
# prediction = knn.predict(X_new)
# print("예측: {}".format(prediction))
# print("예측한 타깃의 이름: {}".format(
#     iris_dataset['target_names'][prediction]))
###

###
# y_pred = knn.predict(X_test)
# print("테스트 세트에 대한 예측값:\n{}".format(y_pred))
# print("테스트 세트의 정확도: {:.2f}".format(numpy.mean(y_pred == y_test)))
###

###
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")
# print("X.shape: {}".format(X.shape))
plt.show()
###

###
# X, y = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X, y, 'o')
# plt.ylim(-3, 3)
# plt.xlabel("특성")
# plt.ylabel("타깃")
# plt.show()
###











#
