import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

print(x.shape)  # (178, 13)


# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=66
# )
n_splits=9
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = SVC()
model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()


#3.4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))


#=============================== StraitifiedKFold 적용 결과 =============================#
# ACC :  [0.9   1.   0.95    0.95    0.9     0.95    0.95    0.94736842   0.94736842]
#  cross_val_score :  0.9439
#==================================== KFold 적용 결과 ===================================#
# ACC :  [1.         0.95       0.95       0.95       1.         0.95
#  0.9        1.         0.94736842]
#  cross_val_score :  0.9608
#=======================================================================================#


