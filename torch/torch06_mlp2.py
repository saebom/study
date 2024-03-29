import numpy as np
import torch
print(torch.__version__)    #1.12.1

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)


#1. 데이터 
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

x_test = np.array([9, 30, 210])
print(x.shape, y.shape, x_test.shape) # (3, 10) (3, 10) (3,)

# '[10, 1.4, 0]의 예측값 : '

############ [실습] 맹그러봐!!! / keras04_2 파이토리로 리폼 ############

x = torch.FloatTensor(x).to(DEVICE)   
y = torch.FloatTensor(y).to(DEVICE)  
x_test = torch.FloatTensor(x_test).unsqueeze(-1).to(DEVICE)  

x = x.T
y = y.T
x_test = x_test.T

print(x.shape, y.shape, x_test.shape)     # torch.Size([10, 2]) torch.Size([10, 1])

# 스케일링
x_test = (x_test - torch.mean(x)) / torch.std(x) 
x = (x - torch.mean(x)) / torch.std(x) 

print(x.shape, y.shape, x_test.shape) # torch.Size([10, 3]) torch.Size([10, 3]) torch.Size([1, 3])


#2. 모델 구성
# model = Sequential()
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 10),
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1)
).to(DEVICE)


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='SGD')
criterion = nn.MSELoss()    # loss
optimizer = optim.SGD(model.parameters(), lr=0.007)    # optimizer
# optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()           # ****훈련모드(디폴트. 명시해주지 않아도 됨)
    optimizer.zero_grad()   # 손실함수의 기울기를 초기화한다
                            # torch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문에 항상 역전파하기 전 gradients를 zero로 만들어주고 시작을 해야 함
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    # loss = nn.MSELoss(hypothesis, y)  # 에러
    # loss = nn.MSELoss()(hypothesis, y)
    # loss = F.mse_loss(hypothesis, y)
    
    loss.backward()     # 역전파
    optimizer.step()     # 가중치를 갱신시키겠다.
    return loss.item()  # .item

epochs = 50
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss: {}'.format(epoch, loss))

    
#4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()    # ****평가모드    

    with torch.no_grad():   # 순전파만 해서 예측함
        y_predict = model(x)
        results = criterion(y_predict, y)
    return results.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

# y_predict = model.predict([4])
results = model(torch.Tensor(x_test).to(DEVICE))

print("[10, 1.4, 0]의 예측값 : ", results.item())

# 최종 loss :  8.271591186523438
# [10, 1.4, 0]의 예측값 :  20.973125457763672