
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_boston, load_diabetes
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
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=2022, shuffle=True
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
print(df)   # alcohol malic_acid   ash alcalinity_of_ash magnesium total_phenols 
            # flavanoids nonflavanoid_phenols proanthocyanins color_intensity   hue od280/od315_of_diluted_wines proline

# df.plot.box()
# plt.title('Wine')
# plt.xlabel('Feature')
# plt.ylabel('Data')
# plt.show()

df['sepal width (cm)'] = np.log1p(df['sepal width (cm)']) 


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
# 기냥 결과 :  1.0
# 로그변환 결과 :  1.0
#==============================================================================#