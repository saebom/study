from sklearn.datasets import load_diabetes
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets['target']

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    shuffle=True, 
    random_state=721
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

print(x_train.size())   # torch.Size([309, 10])

#2. 모델 
# model = nn.Sequential(
#     nn.Linear(10, 32),
#     # nn.Linear(10, 32),
#     # nn.Linear(32, 64),
#     # nn.ReLU(),
#     # nn.Linear(64, 128),
#     # nn.Sigmoid(),
#     # nn.Linear(128, 64),
#     # nn.Sigmoid(),
#     # nn.Linear(64, 32),
#     nn.Linear(32, 1),
# ).to(DEVICE) 

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, output_dim)
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.linear2(x)
        return x
    
model = Model(10, 1).to(DEVICE)    
    
#3. 컴파일, 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-7)
#weight decay : L2 정규화에서 패널티 계수를 의미함. 클수록 제약조건 강함

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()     # 디폴트
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)

    loss.backward()     # 역전파
    optimizer.step()    # 가중치 갱신
    return loss.item()

EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
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
# print(y_predict[:10])

# y_predict = torch.argmax(model(x_test), axis=1)  
print(y_predict[:10])

from sklearn.metrics import r2_score
score = r2_score(y_test.detach().cpu().numpy(), 
                 y_predict.detach().cpu().numpy())
print('r2_score : ', score)

# ========================== 평가, 예측 ============================= #
# loss :  3086.4794921875
# r2_score :  0.48982949594357195
# =================================================================== #