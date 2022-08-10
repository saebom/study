import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])
# print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data.shape)   #(4, 5)
print(data)   

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# SimpleImputer()
# imputer = SimpleImputer()   # 평균값으로 결측치 처리
# imputer = SimpleImputer(strategy='mean')   # 평균값
# imputer = SimpleImputer(strategy='median')   # 중위값
# imputer = SimpleImputer(strategy='most_frequent')   # 가장 빈번히 사용되는 값
# imputer = SimpleImputer(strategy='constant')   # constant는 상수, default는 0
# imputer = SimpleImputer(strategy='constant', fill_value=777)   # constant는 상수, fill_value로 숫자 지정

# KNNImputer()
# imputer = KNNImputer()   # 평균값으로 결측치 처리
# imputer = KNNImputer(n_neighbors=2)   # 근접한 수 입력
# imputer = KNNImputer(n_neighbors=2, weights='uniform', metric='nan_euclidean')

# IterativeImputer()
# imputer = IterativeImputer()
imputer = IterativeImputer(random_state=72)

imputer.fit(data)
data2 = imputer.transform(data)
print(data2)


