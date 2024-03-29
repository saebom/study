from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    shuffle=True, 
    random_state=72
)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())   # torch.Size([14447, 8])

#2. 모델 
model = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(DEVICE) 

#3. 컴파일, 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()     # 디폴트
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)

    loss.backward()     # 역전파
    optimizer.step()    # 가중치 갱신
    return loss.item()

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    running_loss =0.0
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : {}, loss : {:.8f}'.format(epoch, loss))

#4. 평가, 예측
print("========================== 평가, 예측 =============================")
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
    return loss.item()
    
loss2 = evaluate(model, criterion, x_test, y_test)
print('loss : ', loss2)

y_predict = model(x_test)
print(y_predict[:10])

from sklearn.metrics import r2_score
score = r2_score(y_test.detach().cpu().numpy(), 
                 y_predict.detach().cpu().numpy())
print('r2_score : ', score)


# ========================== 평가, 예측 =============================
# loss :  0.42287224531173706
# r2_score :  0.6783785576252674