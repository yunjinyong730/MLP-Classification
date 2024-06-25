# MLP-Classification
MLP를 이용한 분류
# Training Part

```swift
import torch.optim as optim
import numpy as np
from copy import deepcopy # validation 도중 best model 기록하기 위해서 : 원래는 local storage에 저장했다가 학습 이후에 꺼내쓰는데, 이번에는 메모리에 그냥 저장...

optimizer = optim.Adam(params=model.parameters(), lr=0.01) # 활성화 함수는 Adam 사용  Adam(Adagrad + Momentum)
calc_loss = nn.CrossEntropyLoss() # 손실함수는 교차 엔트로피로 구함 <- 소프트맥스로 구하는 거 
n_epoch = 20 # 에폭은 20으로 설정 -> 데이터셋 20번 보겠다!
global_i = 0 # 전체 iteration 수 

#learning curve 그리기 위해!
valid_loss_history = [] # 가중치 validation을 위한 기록 리스트 [(global_i, valid_loss), ...]
train_loss_history = [] 

min_valid_loss = 9e+9 # 최소 손실 값 -> 일단 최대값으로 잡아놓음 
best_model = None # 가장 좋은 모델 RAM에 기록 
best_epoch_i = None # 가장 좋은 모델을 어느 epoch에 봤는지 기록하기 위해 

for epoch_i in range(n_epoch):
    model.train() # 훈련 가능 상태 표시하는 플래그 -> 대표적으로 드롭아웃이 이 flag에 영향을 많이 받음

    for batch in train_dataloader:
        optimizer.zero_grad()
        x = torch.tensor(batch['x'])
        y = torch.tensor(batch['y']) #y.shape: batch_size [0, 1, ...]
        y_pred = model(x) #y_pred.shape: batch_size, output_dim(2)
        loss = calc_loss(y_pred, y)
        
        # epoch마다 필요한 정보 logging
        if global_i % 10 == 0:
            print(f"global_i: {global_i}, epoch_i: {epoch_i}, loss: {loss.item()}")
        train_loss_history.append((global_i, loss.item()))

        loss.backward() # loss를 이용해서 기울기값 backward 오차역전으로 구하기 
        optimizer.step() # optimizer.step()으로 업데이트 
        global_i += 1

    model.eval() # model evalutaion을 하는 이유는 한 번의 epoch이 끝난 다음에 validation 수행 

    #validation
    valid_loss_list = []
    for batch in train_dataloader:
        x = torch.tensor(batch['x'])
        y = torch.tensor(batch['y'])
        y_pred = model(x) # inference 추론 진행 
        loss = calc_loss(y_pred, y) # loss 계산 
        valid_loss_list.append(loss.item()) # valid_loss_list에 저장 

    valid_loss_mean = np.mean(valid_loss_list)
    valid_loss_history.append((global_i, valid_loss_mean.item())) # 나중에 visualization 할 때 사용할 예정 

    if valid_loss_mean < min_valid_loss: # 현재 step의 validation loss mean이 mean validation loss보다 작다면 업데이트 
          min_valid_loss = valid_loss_mean
          best_epoch_i = epoch_i
          best_model = deepcopy(model)

    print("*"*30)
    print(f"valid_loss_mean: {valid_loss_mean}")
    print("*"*30)
    print(f"best_epoch_i: {best_epoch_i}")

```

# Learning Curve
![lossy](https://github.com/yunjinyong730/MLP-Classification/assets/173673702/64f61e0e-70ee-4fbc-8b53-e1492987ad01)
