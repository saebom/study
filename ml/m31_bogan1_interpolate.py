# 결측치 처리
#1. 행 또는 열 삭제
#2. 임의의 값  
#   평균 : mean
#   중위값 : median
#   0 : fillna
#   그 행의 앞의 값 : ffill 
#   그 행의 뒤의 값 : bfill 
#   특정값 : ...
#   기타 등등 ...
#3. 보간 - interpolate => 빈자리를 선형회귀 방식, linear 방식으로 찾아냄
#4. 모델 - predict => 다양한 모델 사용 가능, 결측치에 대한 예측값을 찾아냄
#5. 부스팅계열 - 통상 결측치, 이상치에 대해 자유롭다. 믿거나 말거나 ㅋㅋ
 
import pandas as pd
import numpy as np
from datetime import datetime

dates = ['8/10/2022', '8/11/2022', '8/12/2022', '8/13/2022', '8/14/2022']
dates = pd.to_datetime(dates)
print(dates)

print("====================================")
ts = pd.Series([2, np.nan, np.nan, 8, 10], index=dates)
print(ts)

print("====================================")
ts = ts.interpolate()
print(ts)


# ====================================
# 2022-08-10     2.0
# 2022-08-11     NaN
# 2022-08-12     NaN
# 2022-08-13     8.0
# 2022-08-14    10.0
# dtype: float64
# ====================================
# 2022-08-10     2.0
# 2022-08-11     4.0
# 2022-08-12     6.0
# 2022-08-13     8.0
# 2022-08-14    10.0
# dtype: float64
