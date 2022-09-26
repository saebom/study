from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

#1. 데이터 
path = './_data/torch_data/'
train_dataset = CIFAR10(path, train=True, download=False)
test_dataset = CIFAR10(path, train=False, download=False)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

print(x_train.shape, x_test.size()) # torch.Size([50000, 32, 32, 3]) torch.Size([10000, 32, 32, 3])
print(y_train.shape, y_test.size()) # torch.Size([50000]) torch.Size([10000])

# x_train, x_test = x_train.reshape(50000, 3, 32, 32), x_test.reshape(10000, 3, 32, 32)
x_train, x_test = x_train.permute(0, 3, 2, 1), x_test.permute(0, 3, 2, 1)
print(x_train.shape, x_test.size()) # torch.Size([50000, 3, 32, 32]) torch.Size([10000, 3, 32, 32])

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)

#2. 모델
class CNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.3)
        )
        
        self.hidden_layer4 = nn.Flatten()
        
        self.output_layer = nn.Linear(in_features = 64*2*2, out_features = 10)
    
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = x.view(x.shape[0], -1)  # flatten
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return(x)
    
model = CNN(3).to(DEVICE)

from torchsummary import summary

summary(model, (3, 32, 32))       

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 64, 30, 30]           1,792
#               ReLU-2           [-1, 64, 30, 30]               0
#          MaxPool2d-3           [-1, 64, 15, 15]               0
#            Dropout-4           [-1, 64, 15, 15]               0
#             Conv2d-5          [-1, 128, 13, 13]          73,856
#               ReLU-6          [-1, 128, 13, 13]               0
#          MaxPool2d-7            [-1, 128, 6, 6]               0
#            Dropout-8            [-1, 128, 6, 6]               0
#             Conv2d-9             [-1, 64, 4, 4]          73,792
#              ReLU-10             [-1, 64, 4, 4]               0
#         MaxPool2d-11             [-1, 64, 2, 2]               0
#           Dropout-12             [-1, 64, 2, 2]               0
#           Flatten-13                  [-1, 256]               0
#            Linear-14                   [-1, 10]           2,570
# ================================================================
# Total params: 152,010
# Trainable params: 152,010
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.01
# Forward/backward pass size (MB): 1.52
# Params size (MB): 0.58
# Estimated Total Size (MB): 2.11
# ----------------------------------------------------------------

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
        