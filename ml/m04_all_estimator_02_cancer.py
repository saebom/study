from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        
#3. 훈련


#4. 평가, 예측


#===================================== 결  과 ==========================================#
# AdaBoostClassifier 의 정답률 :  0.9532163742690059
# BaggingClassifier 의 정답률 :  0.9590643274853801
# BernoulliNB 의 정답률 :  0.6432748538011696
# CalibratedClassifierCV 의 정답률 :  0.9824561403508771
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 :  0.8070175438596491
# DecisionTreeClassifier 의 정답률 :  0.9532163742690059
# DummyClassifier 의 정답률 :  0.6432748538011696
# ExtraTreeClassifier 의 정답률 :  0.8947368421052632
# ExtraTreesClassifier 의 정답률 :  0.9532163742690059
# GaussianNB 의 정답률 :  0.9473684210526315
# GaussianProcessClassifier 의 정답률 :  0.9766081871345029
# GradientBoostingClassifier 의 정답률 :  0.9649122807017544
# HistGradientBoostingClassifier 의 정답률 :  0.9707602339181286
# KNeighborsClassifier 의 정답률 :  0.9649122807017544
# LabelPropagation 의 정답률 :  0.9707602339181286
# LabelSpreading 의 정답률 :  0.9707602339181286
# LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
# LinearSVC 의 정답률 :  0.9824561403508771
# LogisticRegression 의 정답률 :  0.9766081871345029
# LogisticRegressionCV 의 정답률 :  0.9766081871345029
# MLPClassifier 의 정답률 :  0.9824561403508771
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 :  0.8596491228070176
# NearestCentroid 의 정답률 :  0.9415204678362573
# NuSVC 의 정답률 :  0.9590643274853801
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.9707602339181286
# Perceptron 의 정답률 :  0.9298245614035088
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9473684210526315
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 :  0.9707602339181286
# RidgeClassifier 의 정답률 :  0.9532163742690059
# RidgeClassifierCV 의 정답률 :  0.9590643274853801
# SGDClassifier 의 정답률 :  0.9473684210526315
# SVC 의 정답률 :  0.9766081871345029
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#

