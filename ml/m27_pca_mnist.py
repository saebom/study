import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist    #tensorflow 2.7 이상부터 공식적으로 keras.datasets 등 keras.으로 권장함

(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)

print(x.shape)  # (70000, 28, 28) => reshape(70000, 28*28)


######################################################################################
#[실습]
# pca를 통해 0.95이상인 n_components는 몇 개?
# 0.95
# 0.99
# 0.999
# 1.0
# 힌트 np.argmax
######################################################################################

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# x = x.reshape(70000, 28*28)
print(x.shape)  # (70000, 784)

pca = PCA(n_components=486)   
                            
x = pca.fit_transform(x)
print(x.shape)  

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR)) 

cumsum = np.cumsum(pca_EVR) 
print(cumsum)

print(np.argmax(cumsum >= 0.95)+1)    # 154
print(np.argmax(cumsum >= 0.99)+1)    # 331
print(np.argmax(cumsum >= 0.999)+1)   # 486
print(np.argmax(cumsum >= 1.0)+1)     # 713


