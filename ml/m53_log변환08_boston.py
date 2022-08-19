
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)     # (506, 13) (506,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=1234, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = LinearRegression()
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print("기냥 결과 : ", round(result, 4)) # 기냥 결과 :  0.9156

############################ 로그 변환 ####################################
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)

# df.plot.box()
# plt.title('Boston')
# plt.xlabel('Feature')
# plt.ylabel('Data')
# plt.show()

print(df['B'].head())
#         B
# 0  396.90
# 1  396.90
# 2  392.83
# 3  394.63
# 4  396.90
df['B'] = np.log1p(df['B']) # np.log1p로 로그 변환함
print(df['B'].head())
#           B
# 0  5.986201
# 1  5.986201
# 2  5.975919
# 3  5.980479
# 4  5.986201

df['CRIM'] = np.log1p(df['CRIM'])   # 0.7596, 0.9186
df['ZN'] = np.log1p(df['ZN'])         # 0.7779, 0.9184
df['TAX'] = np.log1p(df['TAX'])       # 0.7714, 0.9152

x_train, x_test, y_train, y_test = train_test_split(
    df, y, shuffle=True, random_state=1234, train_size=0.8
)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = LinearRegression()
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print("로그변환 결과 : ", round(result, 4)) # 기냥 결과 :  0.9156


#=================================== 결과 =====================================#
# LR 기냥 결과 :  0.9156
# LR 기냥 결과 :  0.7665 => 로그변환 결과 :  0.7711
# RF 기냥 결과 :  0.9125 => 로그변환 결과 :  0.9149
# 'B','ZN', 'TAX' 로그변환 후 LR 결과 :  0.7785
# 'B','ZN', 'TAX' 로그변환 후 RF 결과 :  0.9145
# 'B','CRIM', 'ZN', 'TAX' 로그변환 후 RF 결과 :  0.9201
#==============================================================================#