from keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'
text2 = '나는 지구용사 이재근이다. 멋있다. 또 또 얘기해봐'


token = Tokenizer()
token.fit_on_texts([text1, text2])  # fit하면서 index가 생성됨

print(token.word_index) # 빈도가 높은 index가 앞에 프린트되고 다음 순서대로 프린트 됨
# {'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛있는': 6, '밥을': 7, 
# '엄청': 8, '먹었다': 9, '지구용사': 10, '이재근이다': 11, '멋있다': 12, '얘기해봐': 13}

x = token.texts_to_sequences([text1, text2])
print(x) # [[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]

from tensorflow.python.keras.utils.np_utils import to_categorical

x_new = x[0] + x[1]
print(x_new)
# [2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9, 2, 10, 11, 12, 4, 4, 13]


# x = to_categorical(x_new)
# print(x)
# print(x.shape)

# [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# (18, 14)


############################### 원 핫으로 수정 #############################
from sklearn.preprocessing import OneHotEncoder
import numpy as np

onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
x = np.array(x_new)
print(x.shape)  # (18,)
x = x.reshape(-1, 1)
print(x.shape)  # (18, 1)
onehot_encoder.fit(x)
x = onehot_encoder.transform(x)
print(x)
print(x.shape)  # (18, 13)




