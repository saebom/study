import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
print(x.shape)  # (581012, 54)

# 라벨인코딩
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72, stratify=y
    )

# 스케일링
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)

model = VotingClassifier(
    estimators=[('XG', xg), ('LG', lg), ('CAT', cat)],
    voting='soft',   # hard
    # voting='hard',
    n_jobs=-1
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print('보팅 결과 : ', round(score, 4))

classifiers = [cat, xg, lg]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print('{0} 정확도 : {1:.4f}'.format(class_name, score2), )

#======================== 결과 ========================#
# 보팅 결과 :  0.8777
# CatBoostClassifier 정확도 : 0.8845
# XGBClassifier 정확도 : 0.8691
# LGBMClassifier 정확도 : 0.8573
#======================================================#
