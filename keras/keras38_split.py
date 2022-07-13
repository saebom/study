import numpy as np

a = np.array(range(1, 11)) 
# print(a)   # [ 1  2  3  4  5  6  7  8  9 10]
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):    # 10 - 5 + 1 = 6 :: 리스트 갯수
        subset = dataset[i : (i + size)]         
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)    # (8, 3)

x = bbb[:, :-1]     # x = bbb[:, :4] 동일
y = bbb[:,:-3]       # x = bbb[:, 4] 동일
print(x, y)
print(x.shape, y.shape)     # (9, 1) (9,)