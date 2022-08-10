import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import norm 

x = np.arange(-4, 4, 0.001)
plt.plot(x, norm.pdf(x, loc=0, scale=1))
plt.title("Standard Normal Distribution")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
cum = np.arange(-1, 1, 0.01) #1
plt.fill_between(cum, norm.pdf(cum), alpha=0.5, color='g') #2
pro = norm(0, 1).cdf(1) - norm(0, 1).cdf(-1) #3
plt.text(0, 0.02, round(pro,2), fontsize=20) #4
plt.show()
# 출처: https://bigdata-doctrine.tistory.com/14 [경제와 데이터:티스토리]