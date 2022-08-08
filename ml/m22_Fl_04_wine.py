import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
print(x.shape)  # (178, 13)
# - Alcohol
# - Malic acid
# - Ash
# - Alcalinity of ash
# - Magnesium
# - Total phenols
# - Flavanoids
# - Nonflavanoid phenols
# - Proanthocyanins
# - Color intensity
# - Hue
# - OD280/OD315 of diluted wines
# - Proline


# drop_features
x = np.delete(x, [2, 10], axis=1)
print(x.shape)  # (569, 22)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
    )
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)   # train은 fit_transform, test는 transform으로 overfit(과적합)이 안 잡힘

parameters = [
    {'RF__n_estimators' : [100, 200], 'RF__max_depth':[6, 8, 10, 12], 'RF__n_jobs' : [-1, 2, 4]},  #n_estimators 는 epoch
    # {'RF__max_depth' : [6, 8, 10, 12], 'RF__min_samples_split' : [2, 3, 5, 10]},
    {'RF__n_estimators' : [100, 200], 'RF__min_samples_leaf' : [3, 5, 7, 10]},
    # {'RF__min_samples_split' : [2, 3, 5, 10], 'RF__n_jobs' : [-1, 2, 4]}, 
    # {'RF__n_estimators' : [100, 200],'RF__n_jobs' : [-1, 2, 4]}
]

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=72)


#2. 모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
# print('model.score : ', result) # model.score :  1.0

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test,)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

print('==============================')
print(model, ': ', model.feature_importances_)


#결과비교

#======================================  [2, 10] 삭제  결과 =======================================#
# 1. DecisionTree
# 기존 acc : 0.7777777777777778
# 컬럼 삭제 후 acc :  0.8888888888888888
# [0.02611494 0.         0.         0.         0.03264368 0.   0.05847285 0.         0.         0.35336384 0.071522   0.02074843

# 2. RandomForestClassifier
# 기존 acc : 0.9444444444444444
# 컬럼 삭제 후 acc : 0.9722222222222222
# [0.11996271 0.02585398 0.01064071 0.02422278 0.02045443 0.05146923 0.15486028 0.00840184 0.02794621 0.14126333 0.09113486 0.08839619

# 3. GradientBoostingClassifier
# 기존 acc :  0.8611111111111112
# 컬럼 삭제 후 acc : 0.8611111111111112
# [3.56692535e-03 5.54589820e-02 3.06214081e-05 6.66215209e-03 2.07307903e-04 2.65017167e-05 2.40126298e-01 1.63885935e-04
#  8.59790414e-03 3.40074457e-01 1.96611671e-03 9.30640187e-03 3.33812446e-01]

# 4. XGBClassifier
# 기존 acc : 0.9166666666666666
# 컬럼 삭제 후 acc : 0.9444444444444444
# [0.0679947  0.09255462 0.         0.01077118 0.04155067 0.01352299
#  0.20911653 0.02725588 0.02460833 0.2537367  0.01061006 0.04329838 0.20497994]
#=========================================================================================================================#
