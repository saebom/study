from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
# model.add(Dense(units=10, input_shape=(3,)))   # (batch_size, input_dim)의 형태이므로 input_shape=(10, 10, 3))) 
model.add(Conv2D(filters=10, kernel_size=(3, 3), # 출력 : (4, 4, 10)  (6, 6, 10)  (5, 5, 10) # kernal_size는 이미지를 자르는 규격을 의미  
                 input_shape=(8, 8, 1)))         # (rows, columns, channels)의 형태이므로 None, 5, 5, 1 (1은 흑백, 3은 칼라)
model.add(Conv2D(7, (2, 2), activation='relu'))  # 출력 : (3, 3, 7)   (5, 5, 7)   (4, 4, 7)
model.add(Flatten())    # (N, 63)  (N, 175)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


#=========================================== 내용 정리 ======================================================#
#  OutputShape의 rows, columns수 = (input수 - kernal수) + 1   ===> 성진 교만한 눈빛!!!!!   (-_-);;
#  CNN모델의 output shape의 형태 = (batch_size, rows, colums, channels)로 출력됨
#  Dense모델 Param 갯수 = (input_dim X units) + bias node(unit)
#  CNN모델 Param 갯수 = (kernel_size X channels X filters) + bias node(filter)  ===> 성진 겸손한 눈빛  (0_0)


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 4, 4, 10)          50
# =================================================================
# Total params: 50
# Trainable params: 50
# Non-trainable params: 0


# => 결과 해석 :
# None : 입력될 이미지의 개수는 정해지지 않았으므로 None이며, batch-size가 입력됨
# 4, 4 : MNIST 데이터는 4 x 4 픽셀, kernel_size는 합성곱에 사용되는 필터(=커널)의 크기
# 10 : 10은 채널(channel)을 의미하며, 흑백 이미지 이므로 한 개의 채널을 가지고 필터가 10개이므로 10이 나옴
# CNN모델 Param 갯수 = kernel_size X channels + bias X filters

# 출처: https://excelsior-cjh.tistory.com/152 [EXCELSIOR:티스토리]

#==============================================================================================================#
