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
x = np.array([range(10)])
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])

x_test = np.array([9])
print(x.shape, y.shape, x_test.shape) # (1, 10) (3, 10) (1,)

############ [실습] 맹그러봐 / keras04_mlp3.py 파이토치로 리폼 ############
x = torch.FloatTensor(x).to(DEVICE)   
y = torch.FloatTensor(y).to(DEVICE)  
x_test = torch.FloatTensor(x_test).unsqueeze(-1).to(DEVICE)  

x = x.T
y = y.T
x_test = x_test.T

print(x.shape, y.shape)     # torch.Size([10, 1]) torch.Size([10, 3])

# 스케일링
x_test = (x_test - torch.mean(x)) / torch.std(x) 
x = (x - torch.mean(x)) / torch.std(x) 

print(x.shape, y.shape, x_test.shape) # torch.Size([10, 1]) torch.Size([10, 3]) torch.Size([1, 1])


#2. 모델 구성
# model = Sequential()
model = nn.Sequential(
    nn.Linear(1, 3),
    nn.Linear(3, 6),
    nn.Linear(6, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.Linear(6, 3)
).to(DEVICE)


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='SGD')
criterion = nn.MSELoss()    # loss
optimizer = optim.Adam(model.parameters(), lr=0.01)    # optimizer
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

epochs = 500
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
results = model(x_test).to(DEVICE)

# print("[9]의 예측값 : ", results.detach().cpu().numpy())
print("[9]의 예측값 : ", results.tolist())
# print("[9]의 예측값 : ", results) # 에러발생
# print("[9]의 예측값 : ", results.item())    # 에러발생
# ValueError: only one element tensors can be converted to Python scalars 

# 최종 loss :  2.251236306746729e-13
# [9]의 예측값 :  [[10.0, 1.8999994993209839, 9.5367431640625e-07]]