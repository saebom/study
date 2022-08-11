import numpy as np

aaa = np.array([-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50])
aaa = aaa.reshape(-1, 1)
print(aaa.shape)    # (13, 1)

from sklearn.covariance import EllipticEnvelope
outlier = EllipticEnvelope(contamination=.1)    # contamination=.1 은 이상치를 10%로 잡겠다는 의미

outlier.fit(aaa)
results = outlier.predict(aaa)
print(results)

