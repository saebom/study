import numpy as np

aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]])
aaa = np.transpose(aaa) 
print(aaa.shape)    #(13, 2)
print(aaa)
aaa1 = aaa[:, 0]
aaa2 = aaa[:, -1]

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)    
    lower_bound = quartile_1 - (iqr *1.5)   
    upper_bound = quartile_3 + (iqr *1.5)   
    print("lower_bound : ", lower_bound)
    print("upper_bound : ", upper_bound)
    return np.where((data_out>upper_bound) |
                    (data_out<lower_bound))
    
outliers_loc1 = outliers(aaa1)
print("0열의 이상치의 위치 : ", outliers_loc1)

outliers_loc2 = outliers(aaa2)
print("1열의 이상치의 위치 : ", outliers_loc2)
    
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()    

#========================= 0번째 이상치 =========================#    
# 1사분위 :  4.0
# q2 :  7.0
# 3사분위 :  10.0
# iqr :  6.0
# lower_bound :  -5.0
# upper_bound :  19.0
# 0열의 이상치의 위치 :  (array([ 0, 12], dtype=int64),)

#========================= 1번째 이상치 =========================#    
# 1사분위 :  200.0
# q2 :  400.0
# 3사분위 :  600.0
# iqr :  400.0
# lower_bound :  -400.0
# upper_bound :  1200.0
# 1열의 이상치의 위치 :  (array([6], dtype=int64),)
#================================================================#    

