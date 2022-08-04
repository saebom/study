import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.random.set_seed(66)  # weight에 난수값 

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)   # 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'

x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
   x, y, train_size=0.8, shuffle=True, random_state=72
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
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

# print('allAlgorithms : ', allAlgorithms)    # allAlgorithms의 리스트 출력. key와 value로 이루어진 딕셔너리임.
'''
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
allAlgorithms :  [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), 
('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>), 
('BernoulliNB', <class 'sklearn.naive_bayes.BernoulliNB'>), 
('CalibratedClassifierCV', <class 'sklearn.calibration.CalibratedClassifierCV'>), 
('CategoricalNB', <class 'sklearn.naive_bayes.CategoricalNB'>), 
('ClassifierChain', <class 'sklearn.multioutput.ClassifierChain'>), 
('ComplementNB', <class 'sklearn.naive_bayes.ComplementNB'>), 
('DecisionTreeClassifier', <class 'sklearn.tree._classes.DecisionTreeClassifier'>), 
('DummyClassifier', <class 'sklearn.dummy.DummyClassifier'>), 
('ExtraTreeClassifier', <class 'sklearn.tree._classes.ExtraTreeClassifier'>), g
('ExtraTreesClassifier', <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>), 
('GaussianNB', <class 'sklearn.naive_bayes.GaussianNB'>), 
('GaussianProcessClassifier', <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>), 
('GradientBoostingClassifier', <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>), 
('HistGradientBoostingClassifier', <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>), 
('KNeighborsClassifier', <class 'sklearn.neighbors._classification.KNeighborsClassifier'>), 
('LabelPropagation', <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>), 
('LabelSpreading', <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>), 
('LinearDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>), 
('LinearSVC', <class 'sklearn.svm._classes.LinearSVC'>), ('LogisticRegression', <class 'sklearn.linear_model._logistic.LogisticRegression'>), 
('LogisticRegressionCV', <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>), 
('MLPClassifier', <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>), 
('MultiOutputClassifier', <class 'sklearn.multioutput.MultiOutputClassifier'>), 
('MultinomialNB', <class 'sklearn.naive_bayes.MultinomialNB'>), 
('NearestCentroid', <class 'sklearn.neighbors._nearest_centroid.NearestCentroid'>), 
('NuSVC', <class 'sklearn.svm._classes.NuSVC'>), ('OneVsOneClassifier', <class 'sklearn.multiclass.OneVsOneClassifier'>), 
('OneVsRestClassifier', <class 'sklearn.multiclass.OneVsRestClassifier'>), 
('OutputCodeClassifier', <class 'sklearn.multiclass.OutputCodeClassifier'>), 
('PassiveAggressiveClassifier', <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>), 
('Perceptron', <class 'sklearn.linear_model._perceptron.Perceptron'>), 
('QuadraticDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>), 
('RadiusNeighborsClassifier', <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>), 
('RandomForestClassifier', <class 'sklearn.ensemble._forest.RandomForestClassifier'>), 
('RidgeClassifier', <class 'sklearn.linear_model._ridge.RidgeClassifier'>), 
('RidgeClassifierCV', <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>), 
('SGDClassifier', <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>), 
('SVC', <class 'sklearn.svm._classes.SVC'>), ('StackingClassifier', <class 'sklearn.ensemble._stacking.StackingClassifier'>), 
('VotingClassifier', <class 'sklearn.ensemble._voting.VotingClassifier'>)]
'''
print('모델의 갯수 : ', len(allAlgorithms)) # 41

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

'''    
AdaBoostClassifier 의 정답률 :  1.0
BaggingClassifier 의 정답률 :  1.0
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.9666666666666667
CategoricalNB 의 정답률 :  0.3
ClassifierChain 은 안나온 놈!!!
ComplementNB 의 정답률 :  0.7333333333333333
DecisionTreeClassifier 의 정답률 :  1.0
DummyClassifier 의 정답률 :  0.26666666666666666
ExtraTreeClassifier 의 정답률 :  0.9666666666666667
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  1.0
GradientBoostingClassifier 의 정답률 :  1.0
HistGradientBoostingClassifier 의 정답률 :  1.0
KNeighborsClassifier 의 정답률 :  1.0
LabelPropagation 의 정답률 :  1.0
LabelSpreading 의 정답률 :  1.0
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  1.0
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
MultiOutputClassifier 은 안나온 놈!!!
MultinomialNB 의 정답률 :  0.7333333333333333
NearestCentroid 의 정답률 :  1.0
NuSVC 의 정답률 :  1.0
OneVsOneClassifier 은 안나온 놈!!!
OneVsRestClassifier 은 안나온 놈!!!
OutputCodeClassifier 은 안나온 놈!!!
PassiveAggressiveClassifier 의 정답률 :  0.8666666666666667
Perceptron 의 정답률 :  0.8333333333333334
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.3333333333333333
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.9
RidgeClassifierCV 의 정답률 :  0.9
SGDClassifier 의 정답률 :  0.9
SVC 의 정답률 :  1.0
StackingClassifier 은 안나온 놈!!!
VotingClassifier 은 안나온 놈!!!
'''
    
#3. 훈련


#4. 평가, 예측



#===================================== 결  과 ==========================================#
# LinearSVC() 결과 acc : 1.0
# LogisticRegression() 결과 acc :  1.0
# KNeighborsClassifier() 결과 acc :  0.9666666666666667
# DecisionTreeClassifier() 결과 acc :  1.0
# RandomForestClassifier() 결과 acc :  1.0
#=======================================================================================#


