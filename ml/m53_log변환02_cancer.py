
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_boston, load_diabetes
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
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)     # (569, 30) (569,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = LinearRegression()
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print("기냥 결과 : ", round(result, 4)) # 기냥 결과 :  0.9999

############################ 로그 변환 ####################################
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)   # mean radius mean texture mean perimeter mean area mean smoothness mean compactness mean concavity  
# ... worst area worst smoothness worst compactness worst concavity worst concave points worst symmetry worst fractal dimension

# df.plot.box()
# plt.title('Cancer')
# plt.xlabel('Feature')
# plt.ylabel('Data')
# plt.rc('xtick', labelsize=1)  # x축 눈금 폰트 크기 
# plt.show()

df['mean perimeter'] = np.log1p(df['mean perimeter']) 
df['mean area'] = np.log1p(df['mean area']) 
df['perimeter error'] = np.log1p(df['perimeter error']) 
df['area error'] = np.log1p(df['area error']) 
df['worst texture'] = np.log1p(df['worst texture']) 
df['worst perimeter'] = np.log1p(df['worst perimeter']) 


x_train, x_test, y_train, y_test = train_test_split(
    df, y, shuffle=True, random_state=72, train_size=0.8
)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = LinearRegression()
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print("로그변환 결과 : ", round(result, 4)) # 기냥 결과 :  0.9156


#=================================== 결과 =====================================#
# 기냥 결과 :  0.8506
# 로그변환 결과 :  0.888
#==============================================================================#