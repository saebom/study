import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

#1. 데이터
path = 'D:/study_data/_data/'
data_set = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';')

print('data_set.shape', data_set.shape)    # (4898, 12)
print(data_set.describe())
print(data_set.info())

# x = data_set.drop(['quality'], axis=1)
# y = data_set['quality']

data_set2 = data_set.to_numpy()
# data_set2 = data_set.values
print(type(data_set2))
print(data_set2.shape)  # (4898, 12)

x = data_set2[:, :11]
y = data_set2[:, 11]
print(x.shape, y.shape) # (4898, 11) (4898,)
print(np.unique(y, return_counts=True))  
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(data_set['quality'].value_counts)
# print(y[:10])   # [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
# print(y[1000:1010]) # [7. 5. 6. 7. 6. 6. 6. 5. 7. 6.]
print(y[:20])   # [6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 5. 5. 5. 7. 5. 7. 6. 8. 6. 5.]

newlist = []
for i in y:
    if i <=4: 
        newlist += [0]
    elif i <=6:
        newlist += [1]
    else:
        newlist += [2]

print(np.unique(newlist, return_counts=True))
# (array([0, 1, 2]), array([ 183, 3655, 1060], dtype=int64))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(
    x, newlist, random_state=123, shuffle=True, train_size=0.8,
    stratify=y
)
print(pd.Series(y_train).value_counts())
# 1    2924
# 2     848
# 0     146
print("#========================== SMOTE 적용 후 ============================ ")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123, k_neighbors=1)  
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())
# 0    2924
# 1    2924
# 2    2924

#2. 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=1000, max_depth=100))    
# model = RandomForestClassifier()

#3. 훈련

model.fit(x_train, y_train)
print(pd.Series(y_train).value_counts())


#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score

y_predict = model.predict(x_test) 
score = model.score(x_test, y_test)
print('model.score : ', score)     
print('acc_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro'))   # f1_score는 이진분류이므로 average를 사용하여 다중분류에 사용
print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro'))  



#=============================== describe() ==================================#

#        fixed acidity  volatile acidity  citric acid  residual sugar    chlorides  free sulfur dioxide  total sulfur dioxide      density           pH    sulphates      alcohol      quality
# count    4898.000000       4898.000000  4898.000000     4898.000000  4898.000000          4898.000000           4898.000000  4898.000000  4898.000000  4898.000000  4898.000000  4898.000000
# mean        6.854788          0.278241     0.334192        6.391415     0.045772            35.308085            138.360657     0.994027     3.188267     0.489847    10.514267     5.877909
# std         0.843868          0.100795     0.121020        5.072058     0.021848            17.007137             42.498065     0.002991     0.151001     0.114126     1.230621     0.885639
# min         3.800000          0.080000     0.000000        0.600000     0.009000             2.000000              9.000000     0.987110     2.720000     0.220000     8.000000     3.000000
# 25%         6.300000          0.210000     0.270000        1.700000     0.036000            23.000000            108.000000     0.991723     3.090000     0.410000     9.500000     5.000000
# 50%         6.800000          0.260000     0.320000        5.200000     0.043000            34.000000            134.000000     0.993740     3.180000     0.470000    10.400000     6.000000
# 75%         7.300000          0.320000     0.390000        9.900000     0.050000            46.000000            167.000000     0.996100     3.280000     0.550000    11.400000     6.000000
# max        14.200000          1.100000     1.660000       65.800000     0.346000           289.000000            440.000000     1.038980     3.820000     1.080000    14.200000     9.000000


#================================= info() ====================================#
# Data columns (total 12 columns):
#  #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   fixed acidity         4898 non-null   float64
#  1   volatile acidity      4898 non-null   float64
#  2   citric acid           4898 non-null   float64
#  3   residual sugar        4898 non-null   float64
#  4   chlorides             4898 non-null   float64
#  5   free sulfur dioxide   4898 non-null   float64
#  6   total sulfur dioxide  4898 non-null   float64
#  7   density               4898 non-null   float64
#  8   pH                    4898 non-null   float64
#  9   sulphates             4898 non-null   float64
#  10  alcohol               4898 non-null   float64
#  11  quality               4898 non-null   int64
 
#================================== 결 과 =====================================#
# model.score :  0.7316326530612245
# acc_score :  0.7316326530612245
# f1_score(macro) :  0.4436222951519518
# f1_score(micro) :  0.7316326530612245
# ==========================================
# 라벨 축소 후
# model.score :  0.8744897959183674
# acc_score :  0.8744897959183674
# f1_score(macro) :  0.6377678308865878
# f1_score(micro) :  0.8744897959183674
# ==========================================
# SMOTE 적용 후(k_neighbors=1)
# model.score :  0.8561224489795919
# acc_score :  0.8561224489795919
# f1_score(macro) :  0.7079527519081245
# f1_score(micro) :  0.8561224489795919
#==============================================================================#
