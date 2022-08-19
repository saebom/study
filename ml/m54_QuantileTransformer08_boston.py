
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

# 스케일링
# scaler = StandardScaler()               #  0.7665
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MinMaxScaler()                   #  0.7665
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)   #  0.7665
# x_test = scaler.transform(x_test)

# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)   #  0.7665
# x_test = scaler.transform(x_test)

# scaler = QuantileTransformer()            #  0.7665
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = PowerTransformer(method='yeo-johnson') # 0.8022 (method = 'yeo-johnson'이 default)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = PowerTransformer(method='box-cox')       # ValueError: The Box-Cox transformation can only be applied to strictly positive data
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# for 문으로 한번에 확인하기
sts = StandardScaler() 
mms = MinMaxScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
qtf = QuantileTransformer() 
ptf1 = PowerTransformer(method='yeo-johnson')
ptf2 = PowerTransformer(method='box-cox')

scalers = [sts, mms, mas, rbs, qtf, ptf1]
for scaler in scalers:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # model = LinearRegression()
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    result = r2_score(y_test, y_predict)
    scale_name = scaler.__class__.__name__
    print('{0} 결과 : {1:.4f}'.format(scale_name, result), )
    
    
#2. 모델
model = LinearRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print("기냥 결과 : ", round(result, 4)) # 기냥 결과 :  0.9156

'''
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
'''

#=================================== 결과 =====================================#
# StandardScaler 결과 : 0.7665
# MinMaxScaler 결과 : 0.7665
# MaxAbsScaler 결과 : 0.7665
# RobustScaler 결과 : 0.7665
# QuantileTransformer 결과 : 0.7607
# PowerTransformer 결과 : 0.7611
# 기냥 결과 :  0.7611
# ===========================================
# StandardScaler 결과 : 0.9219
# MinMaxScaler 결과 : 0.9165
# MaxAbsScaler 결과 : 0.9122
# RobustScaler 결과 : 0.9100
# QuantileTransformer 결과 : 0.9130
# PowerTransformer 결과 : 0.9176
#==============================================================================#