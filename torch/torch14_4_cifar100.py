from sqlite3 import DatabaseError
from torchvision.datasets import CIFAR100
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cup')

#1. 데이터
path = './_data/torch_data/'
train_dataset = CIFAR100(path, train=True, download=False)
test_dataset = CIFAR100(path, train=False, download=False)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

print(x_train.shape, x_test.size()) # torch.Size([50000, 32, 32, 3]) torch.Size([10000, 32, 32, 3])
print(y_train.shape, y_test.size()) # torch.Size([50000]) torch.Size([10000])

x_train, x_test = x_train.reshape(50000, 32*32*3), x_test.reshape(10000, 32*32*3)
print(x_train.shape, x_test.size()) # torch.Size([50000, 3072]) torch.Size([10000, 3072])

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)

#2. 모델
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(128, 254),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(254, 100),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Linear(100, 100)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return(x)
    
model = DNN(x_train.shape[1]).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr = 1e-4)

def train(model, criterion, optimizer, loader):
    
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)    
# hist = model.fit(x_train, y_train) # hist에는 loss와 acc가 들어가
# 엄밀히 얘기하면 hist라고 하기는 그렇고, loss와 acc를 반환해준다고 혀!!!

#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()    # .eval() 함수는 dropout, batchnormalization 등 
                    # 모델의 레이어 상에서 적용한 것들을 진행하지 않는다
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            
            loss = criterion(hypothesis, y_batch)
               
            epoch_loss += loss.item()
            
            y_predict = torch.argmax(hypothesis, 1)        
            acc = (y_predict == y_batch).float().mean()
            
            epoch_acc += acc.item()
        return epoch_loss / len(loader), epoch_acc / len(loader)     
# loss, acc = model.evaluate(x_test, y_test)        

epochs = 20
for epoch in range(1, epochs +1):
    
    loss, acc = train(model, criterion, optimizer, train_loader)    
    
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print('epoch:{}, loss:{:.4f}, acc:{:.3f}, val_loss:{:.4f}, val_acc:{:.3f}'.format(
        epoch, loss, acc, val_loss, val_acc
    ))

    


