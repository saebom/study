from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

print(x.shape)  # (178, 13)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

#2. 모델 구성, 훈련, 평가, 예측
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
        

#===================================== 결  과 ==========================================#
# AdaBoostClassifier 의 정답률 :  0.5370370370370371
# BaggingClassifier 의 정답률 :  0.9629629629629629
# BernoulliNB 의 정답률 :  0.4074074074074074
# CalibratedClassifierCV 의 정답률 :  0.9444444444444444
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 :  0.7407407407407407
# DecisionTreeClassifier 의 정답률 :  0.9444444444444444
# DummyClassifier 의 정답률 :  0.4074074074074074
# ExtraTreeClassifier 의 정답률 :  0.8518518518518519
# ExtraTreesClassifier 의 정답률 :  0.9814814814814815
# GaussianNB 의 정답률 :  0.9814814814814815
# GaussianProcessClassifier 의 정답률 :  0.37037037037037035
# GradientBoostingClassifier 의 정답률 :  0.9629629629629629
# HistGradientBoostingClassifier 의 정답률 :  1.0
# KNeighborsClassifier 의 정답률 :  0.6851851851851852
# LabelPropagation 의 정답률 :  0.5185185185185185
# LabelSpreading 의 정답률 :  0.5185185185185185
# LinearDiscriminantAnalysis 의 정답률 :  0.9814814814814815
# LinearSVC 의 정답률 :  0.6296296296296297
# LogisticRegression 의 정답률 :  0.9629629629629629
# LogisticRegressionCV 의 정답률 :  0.9444444444444444
# MLPClassifier 의 정답률 :  0.48148148148148145
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 :  0.8333333333333334
# NearestCentroid 의 정답률 :  0.6851851851851852
# NuSVC 의 정답률 :  0.9444444444444444
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.7037037037037037
# Perceptron 의 정답률 :  0.7407407407407407
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9629629629629629
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 :  1.0
# RidgeClassifier 의 정답률 :  0.9814814814814815
# RidgeClassifierCV 의 정답률 :  0.9814814814814815
# SGDClassifier 의 정답률 :  0.7222222222222222
# SVC 의 정답률 :  0.6666666666666666
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#


