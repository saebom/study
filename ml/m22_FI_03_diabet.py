import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(datasets.DESCR)
print(x.shape)  # (442, 10)

# drop_features
x = np.delete(x, [0, 1, 7], axis=1)
print(x.shape)  # (442, 7)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
    )
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)   # train은 fit_transform, test는 transform으로 overfit(과적합)이 안 잡힘


#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()


#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
print('model.score : ', result) # model.score :  1.0

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('r2_score : ', acc)



#======================================  [age, sex, s4 ] 삭제  결과 =======================================#
# 1. DecisionTree
# 기존 r2 : 0.0931371643680513
# 컬럼 삭제 후 r2 :  0.17272382158077348

# 2. RandomForestClassifier
# 기존 r2 : 0.5508741236890558
# 컬럼 삭제 후 r2 : 0.5019665005381797

# 3. GradientBoostingClassifier
# 기존 r2 :   0.5236608835304826
# 컬럼 삭제 후 r2 : 0.516525511785247

# 4. XGBClassifier
# 기존 r2 : 0.5149645205811251
# 컬럼 삭제 후 r2 : 0.39577733354774336
#=========================================================================================================================#

