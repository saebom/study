import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_breast_cancer()

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)

#2. 모델
kmeans = KMeans(n_clusters=2, random_state=1004)

#3. 훈련
kmeans.fit(df)

#4. 평가, 예측
print(kmeans.labels_)
print(datasets.target)

df['cluster'] = kmeans.labels_
df['target'] = datasets.target

result = accuracy_score(df['cluster'], df['target'])
print('acc : ', result)

# acc :  0.8541300527240774