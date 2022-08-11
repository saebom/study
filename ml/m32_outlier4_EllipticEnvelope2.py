import numpy as np

aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]])
aaa = np.transpose(aaa) 
print(aaa.shape)    #(13, 2)
print(aaa)

aaa1 = aaa[:, 0]
aaa2 = aaa[:, -1]
aaa1 = aaa1.reshape(-1, 1)
aaa2 = aaa2.reshape(-1, 1)

from sklearn.covariance import EllipticEnvelope
outlier = EllipticEnvelope(contamination=.1)    # contamination=.1 은 이상치를 10%로 잡겠다는 의미

outlier.fit(aaa1)
results1 = outlier.predict(aaa1)
print(results1)

outlier.fit(aaa2)
results2 = outlier.predict(aaa2)
print(results2)


#==================================== 결과 ==================================#
# contamination=.1일 때
# [-1  1  1  1  1  1  1  1  1  1  1  1 -1]
# [ 1  1  1  1  1  1 -1  1  1 -1  1  1  1]
#============================================================================#
