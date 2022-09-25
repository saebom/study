from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    shuffle=True, 
    random_state=123,
    stratify=y
)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE) # 에러 new(): data must be a sequence (got StandardScaler)

print(x_train.size())   # torch.Size([398, 30])

################################# 요기서 시작 ##################################
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train) # x와 y를 합친다!!!
test_set = TensorDataset(x_test, y_test) # x와 y를 합친다!!!

print(train_set)
print('==========================train_set[0]==============================')
print(train_set[0])
print('=========================train_set[0][0]==============================')
print(train_set[0][0])
print('=========================train_set[0][1]==============================')
print(train_set[0][1])
print('=========================train_set[0][1]==============================')
print(len(train_set))   # 398


# x, y 배치 합체!!! 두그둥!!! 
train_loader = DataLoader(train_set, batch_size=40, shuffle=True)   # DataLoader에서 batch를 정의함
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)     # DataLoader에서 batch를 정의함

print(train_loader) # <torch.utils.data.dataloader.DataLoader object at 0x000001840A81D910>
# print('==========================train_loader[0]==============================')
# print(train_loader[0])  # 'DataLoader' object is not subscriptable
print(len(train_loader))   # 10


#2. 모델
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1),
#     nn.Sigmoid()
# ).to(DEVICE) 
# 클래스 구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__()  # 상위 클래스의 특성을 불러옴
        super(Model, self).__init__() # 동일
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
        
# model = Model(30, 1).to(DEVICE)        

model2 = Model(30, 1).to(DEVICE)

path = './_save/'

##################################### 세이브 로드 #########################################
# torch.save(model.state_dict(), path +'torch13_state_dict.pt')   # 토치의 가중치와 모델이 저장됨
model2.load_state_dict(torch.load(path + 'torch13_state_dict.pt'))
###########################################################################################

##################################### 하단거 붙임 ##########################################
y_predict = (model2(x_test) >= 0.5).float()  # .float()은 boolean이 실수형으로 나옴
print(y_predict[:10])

score = (y_predict == y_test).float().mean() 
print('accuracy :{:.4f}'.format(score))
 
from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict)   # 에러
score = accuracy_score(y_test.cpu(), y_predict.cpu())
print('accuracy_score : ', score)

'''
#3. 컴파일, 훈련
criterion = nn.BCELoss() #binary_crossentropy loss

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train()     # 디폴트
    total_loss = 0
    for x_batch, y_batch in loader:    
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()     # 역전파
        optimizer.step()    # 가중치 갱신
        total_loss += loss.item()
        
    return total_loss / len(loader)

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch : {}, loss : {:.8f}'.format(epoch, loss))

#4. 평가, 예측
print("========================== 평가, 예측 =============================")
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0 
    
    for x_batch, y_batch in loader:        
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
            
        return total_loss
    
loss2 = evaluate(model, criterion, test_loader)
print('loss : ', loss2)

# y_predict = model(x_test)
# print(y_predict[:10])

y_predict = (model(x_test) >= 0.5).float()  # .float()은 boolean이 실수형으로 나옴
print(y_predict[:10])

score = (y_predict == y_test).float().mean() 
print('accuracy :{:.4f}'.format(score))
 
from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict)   # 에러
score = accuracy_score(y_test.cpu(), y_predict.cpu())
print('accuracy_score : ', score)
'''

# ========================== 평가, 예측 =============================
# loss :  0.634477436542511
# accuracy :0.9825
# accuracy_score :  0.9824561403508771
