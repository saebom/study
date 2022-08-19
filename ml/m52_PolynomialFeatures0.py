from cgi import print_exception
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

#1. 데이터
x = np.arange(8).reshape(4, 2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
print(x.shape)  #(4, 2)

pf = PolynomialFeatures(degree=2)
x_df = pf.fit_transform(x)
print(x_df)
# degree=2
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
print(x_df.shape)   # (4, 6)


##################################################################

x = np.arange(12).reshape(4, 3)
print(x)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
print(x.shape)  #(4, 2)

pf = PolynomialFeatures(degree=2)
x_df = pf.fit_transform(x)
print(x_df)
# degree=2
# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]
print(x_df.shape)   # (4, 6)


#######################################################################################
# 기존의 파이프라인에 PCA/ PolnomaialFeatures/ Kmeans 를 붙여서 쓸 수 있다.