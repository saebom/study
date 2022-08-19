
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
datasets = load_diabetes()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)     # (506, 13) (506,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8
)

sts = StandardScaler() 
mms = MinMaxScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
qtf = QuantileTransformer() 
ptf1 = PowerTransformer(method='yeo-johnson')
ptf2 = PowerTransformer(method='box-cox')

scalers = [sts, mms, mas, rbs, qtf, ptf1, ptf2]
for scaler in scalers:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = LinearRegression()
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
print("기냥 결과 : ", round(result, 4)) 


#=================================== 결과 =====================================#
# StandardScaler 결과 : 0.6579
# MinMaxScaler 결과 : 0.6579
# MaxAbsScaler 결과 : 0.6579
# RobustScaler 결과 : 0.6579
# QuantileTransformer 결과 : 0.6490
# PowerTransformer 결과 : 0.6446
# ValueError: The Box-Cox transformation can only be applied to strictly positive data
#==============================================================================#