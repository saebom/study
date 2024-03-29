from keras.preprocessing.text import Tokenizer
import numpy as np
import tensorflow as tf
print(tf.__version__)   # 2.10.0-dev20220713

        
#1. 데이터
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '민수가 못 생기긴 했어요',
        '안결 혼해요'
        ]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])    #()

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

# {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, 
# '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, 
# '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, 
# '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, 
# '재미없다': 23, '재밌네요': 24, '민수가': 25, '못': 26, '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], 
# [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]] 
# => 길이가 서로 다르므로 가장 긴 것의 길이에 맞춰줌

# from keras.preprocessing.sequence import pad_sequences    => ㅠㅠ keras.preprocessing.sequence 에 pad_sequences 없음
from keras_preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre', maxlen=5)
print(pad_x)
print(pad_x.shape)  #(14, 5)

word_size = len(token.word_index)
print("word_size :", word_size) #단어사전의 갯수 : 30

print(np.unique(pad_x, return_counts=True))

#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding

model = Sequential()
                    # input_dim=31은 단어사전의 갯수 (30 + 1(padding 숫자)) 
                    # => padding으로 0을 채워주면서 1부터 있는 index 숫자에 0이 포함 됨, 그래서 +1의 수가 됨
                    # input은 (14, 5), output은 10,
                    # input_length는 명시해도 되고 안해도 됨(input_length는 열의 갯수이며 알고 있을 때 입력의 자율성이 있음)
# model.add(Embedding(input_dim=31, output_dim=10, input_length=5)) # OneHotEncoding을 하지 않고 Embedding layer로 연산함, 단어의 상관관계에 대한 연산
# model.add(Embedding(input_dim=31, output_dim=10))           
# model.add(Embedding(31, 10))    #단어사전의 갯수는 31, output 갯수는 10개
# model.add(Embedding(31, 10, 5))    #ValueError: Could not interpret initializer identifier: 5
model.add(Embedding(30, 10, input_length=5))    #따라서 input_lengh는 명시해주어야 함
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=20, batch_size=16)


#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]  # [0]이면 loss, [1]이면 accuracy 
print('acc : ', acc)


#=================================================================================
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 3)              93
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                4608
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 4,734
# Trainable params: 4,734
# Non-trainable params: 0
# ___________________________

# model.add(Embedding(31, 3, input_length=5))

