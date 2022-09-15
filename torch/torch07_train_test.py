import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])

y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

#### [11, 12, 13]을 예측하라!!!
x_predict = np.array([11, 12, 13])

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (7,) (7,) (3,) (3,)

# 시작!!!!
x_train = torch.FloatTensor(x_train).unsqueeze(-1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(-1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)
x_predict = torch.FloatTensor(x_predict).unsqueeze(-1).to(DEVICE)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_predict) # (7,) (7,) (3,) (3,)

# 스케일링
x_predict = (x_predict - torch.mean(x_train)) / torch.std(x_train)
x_test = (x_test - torch.mean(x_train)) / torch.std(x_train)
x_train = (x_train - torch.mean(x_train)) / torch.std(x_train)

#2. 모델 구성
model = nn.Sequential(
    nn.Linear(1,3),
    nn.Linear(3,6),
    nn.ReLU(),
    nn.Linear(6,12),
    nn.ReLU(),
    nn.Linear(12,6),
    nn.ReLU(),
    nn.Linear(6,1)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

def train(model, criterion, optimizer, x_train, y_train):
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : {}, loss : {}'.format(epoch, loss))
    
#4. 평가, 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x_test)
        results = criterion(y_predict, y_test)
    return results.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)

results = model(x_predict).to(DEVICE)
print("[11, 12, 13]의 예측값 : ", results.detach().cpu().numpy())

# [11, 12, 13]의 예측값 :  [[11.072411 ], [12.07937  ], [13.0863285]]
