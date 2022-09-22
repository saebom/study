from sklearn.datasets import load_wine
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

x = torch.FloatTensor(x)
y = torch.LongTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    shuffle=True, 
    random_state=72,
    stratify=y
)

x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.size())   #torch.Size([124, 13])

############################## DataLoader 시작 #################################
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)


#2. 모델
# model = nn.Sequential(
#     nn.Linear(13, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 3),
#     # nn.Softmax(),
# ).to(DEVICE) 
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x

model = Model(13, 3).to(DEVICE)

#3. 컴파일, 훈련
# criterion = nn.BCELoss() #binary_crossentropy loss
criterion = nn.CrossEntropyLoss() #crossentropy loss

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
    
loss2 = evaluate(model, criterion,test_loader)
print('loss : ', loss2)

y_predict = torch.argmax(model(x_test), axis=1)  

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict)   # 에러
score = accuracy_score(y_test.cpu(), y_predict.cpu())
print('accuracy_score : ', score)

# ========================== 평가, 예측 =============================
# loss :  0.7627812027931213
# accuracy_score :  0.6481481481481481