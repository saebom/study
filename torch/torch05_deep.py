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
x = np.array([1,2,3])   # (3, )
y = np.array([1,2,3])
x_test = np.array([4])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)   # gpu 형태로 사용
y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)  # gpu 형태로 사용
x_test = torch.FloatTensor(x_test).unsqueeze(-1).to(DEVICE)  # gpu 형태로 사용

##################### scaling ######################
x_test = (x_test - torch.mean(x)) / torch.std(x) 
x = (x - torch.mean(x)) / torch.std(x) 
####################################################

print(x, y)
print(x.shape, y.shape) # torch.Size([3, 1]) torch.Size([3, 1])


#2. 모델 구성
# model = Sequential()
model = nn.Sequential(
    nn.Linear(1, 4),
    nn.Linear(4, 5),
    nn.Linear(5, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1)
).to(DEVICE)


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='SGD')
criterion = nn.MSELoss()    # loss
optimizer = optim.SGD(model.parameters(), lr=0.1)    # optimizer
# optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()           # ****훈련모드(디폴트. 명시해주지 않아도 됨)
    optimizer.zero_grad()   # 손실함수의 기울기를 초기화한다
                            # torch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문에 항상 역전파하기 전 gradients를 zero로 만들어주고 시작을 해야 함
    hypothesis = model(x)
    
    # loss = criterion(hypothesis, y)
    # loss = nn.MSELoss(hypothesis, y)  # 에러
    # loss = nn.MSELoss()(hypothesis, y)
    loss = F.mse_loss(hypothesis, y)
    
    loss.backward()     # 역전파
    optimizer.step()      # 가중치를 갱신시키겠다.
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

print("4의 예측값 : ", results.item())

# 최종 loss :  3.805193313222155e-11
# 4의 예측값 :  4.000011444091797