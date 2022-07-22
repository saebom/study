from keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'

token = Tokenizer()
token.fit_on_texts([text])  # fit하면서 index가 생성됨

print(token.word_index) # 빈도가 높은 index가 앞에 프린트되고 다음 순서대로 프린트 됨
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}

x = token.texts_to_sequences([text])
print(x) # [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

x = to_categorical(x)
print(x)
print(x.shape)

# [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]
# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# (1, 11, 9)


############################### 원 핫으로 수정 #############################
# onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
# x = x.reshape(-1, 11, 9)
# onehot_encoder.fit(x)
# x = onehot_encoder.transform(x)





