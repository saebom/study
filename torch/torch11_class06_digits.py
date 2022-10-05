from sklearn.datasets import load_digits
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0

#1. 데이터
datasets = load_digits()
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

print(x_train.size())   # torch.Size([1257, 64])

#2. 모델 
# model = nn.Sequential(
#     nn.Linear(64, 128),
#     nn.ReLU(),
#     nn.Linear(128, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 10),
#     # nn.Softmax(),
# ).to(DEVICE) 
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
    
model = Model(64, 10).to(DEVICE)


#3. 컴파일, 훈련
# criterion = nn.BCELoss() #binary_crossentropy loss
criterion = nn.CrossEntropyLoss() #crossentropy loss

optimizer = optim.Adam(model.parameters(), lr=0.01)

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

# y_predict = model(x_test)
# print(y_predict[:10])

y_predict = torch.argmax(model(x_test), axis=1)  

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict)   # 에러
score = accuracy_score(y_test.cpu(), y_predict.cpu())
print('accuracy_score : ', score)

# ========================== 평가, 예측 =============================
# loss :  0.16481338441371918
# accuracy_score :  0.9740740740740741
# =================================================================== #
