from keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000                     # num_words는 단어의 빈도수 설정, embedding의 input_dim에 넣어주면 됨
)

print(x_train)
print(x_train.shape, x_test.shape)    #(25000,) (25000,) => 25000개씩의 리스트  
print(y_train)    
print(np.unique(y_train, return_counts=True))    
print(y_train.shape, y_test.shape)    #(25000,) (25000,) => 25000개씩의 리스트 
print(len(np.unique(y_train)))        # 2개의 label이 있음, 다중분류 

print(type(x_train), type(y_train))   # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))               # <class 'list'>
# print(x_train[0].shape)             # AttributeError: 'list' object has no attribute 'shape'
print(len(x_train[0]))                # 218
print(len(x_train[1]))                # 189

# print(len(max(x_train)))
print('리뷰의 최대길이 : ', max(len(i) for i in x_train))   # 리뷰의 최대길이 :  2494
print('리뷰의 평균길이 : ', sum(map(len, x_train)) / len(x_train))  # 리뷰의 평균길이 :  238.71364

#전처리
from keras_preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
                        #shape=(25000,) => (25000, 100)
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

y_train = to_categorical(y_train)                        
y_test = to_categorical(y_test)                        

print(x_train.shape, y_train.shape)     # (25000, 100) (25000, 2)
print(x_test.shape, y_test.shape)       # (25000, 100) (25000, 2)


#2. 모델 구성
# [실습]
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Dropout

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=512, validation_split=0.2)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)[0]  # [0]이면 loss, [1]이면 accuracy 
print('loss : ', loss)

acc = model.evaluate(x_test, y_test)[1]  # [0]이면 loss, [1]이면 accuracy 
print('acc : ', acc)

y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)      
y_test = y_test.argmax(axis=1)


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, None, 100)         1000000
# _________________________________________________________________
# lstm (LSTM)                  (None, 128)               117248
# _________________________________________________________________
# dropout (Dropout)            (None, 128)               0
# _________________________________________________________________
# dense (Dense)                (None, 64)                8256
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                4160
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 64)                0
# _________________________________________________________________
# dense_2 (Dense)              (None, 64)                4160
# _________________________________________________________________
# dense_3 (Dense)              (None, 46)                2990
# =================================================================
# Total params: 1,136,814
# Trainable params: 1,136,814
# Non-trainable params: 0

# ============================================ 결과  ===============================================
# loss :  1.1077522039413452
# acc :  0.7095999717712402
# ==================================================================================================
    

