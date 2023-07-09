import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

def training(model, optimizer, train_data, test_data, epoch,device):
    for ep in range(epoch):
        train_loss = 0.
        test_loss = 0.
        test_acc = 0.
        total_loss = 0.
        iteration = 0
        loss_func = nn.MSELoss()
        print("epoch: {}".format(ep))
        for (x,t) in train_data:
            print("iteration: {}".format(iteration))
            t = torch.eye(10)[t]
            model.train()
            optimizer.zero_grad()
            # print(x.shape)
            x,t = x.to(device), t.to(device)
            pred = model(x)
            loss = loss_func(pred,t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iteration+=1
            print("iteration_loss: {}".format(loss.item()))
        train_loss = total_loss / len(train_data)
        print("epoch_loss: {}".format(train_loss))

        total_loss = 0.
        pos = 0
        for (x,t) in test_data:
            model.eval()
            t_l = torch.eye(10)[t]
            x,t,t_l = x.to(device), t.to(device), t_l.to(device)
            pred = model(x)
            loss = loss_func(pred,t_l)
            total_loss += loss.item()
            pred = pred.argmax(1)
            pos += torch.eq(pred,t).sum().item() / len(t)

        test_loss = total_loss / len(test_data)
        test_acc = pos / len(test_data)
        print("test_acc: {}".format(test_acc))

    return train_loss, test_acc 
