import os
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

from model.MyMoblenet2 import MyMobileNetv2
from model.data import ImageFolder


def update_lr(old_lr, new_lr, mylog, factor=False):
    if factor:
        new_lr = old_lr / new_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    print('update learning rate:', old_lr, new_lr, file=mylog)
    print('update learning rate:', old_lr, new_lr)
    old_lr = new_lr

    return old_lr


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 可用范围0~3

ROOT = './dataset/crop_face/Trainset/'
imagelist = filter(lambda x: x.find('IMG') != -1, os.listdir(ROOT))
trainlist = list(imagelist)
print(len(trainlist))
label = pd.read_csv('./dataset/samples.csv')
NAME = 'spacial_model'
batchsize = 1

dataset = ImageFolder(trainlist, ROOT, label)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True)

print('log')
mylog = open('./logs/' + NAME + '.log', 'w')
solver = MyMobileNetv2()
tic = time()
no_optim = 0
total_epoch = 100
train_epoch_best_loss = 100
LEARNING_RATE = 1e-5
old_lr = LEARNING_RATE
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(solver.parameters(), lr=LEARNING_RATE)
print('training')
# //////////////////////////
alpha = 0.9
for epoch in range(1, total_epoch + 1):
    '''////////////
    alpha = epoch/(total_epoch)
    if alpha > 0.8:
        alpha = 0.8
    '''
    jishu = 0
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0

    loss1_sum = 0
    print('epoch:', epoch, len(data_loader_iter))
    loss_log = []
    correct = 0
    if epoch < 10:
        savepath = './logs/loss/loss_epoch_0' + str(epoch) + '.txt'
    else:
        savepath = './logs/loss/loss_epoch_' + str(epoch) + '.txt'
    for videoseg, mask in data_loader_iter:
        videoseg = torch.squeeze(videoseg)
        jishu += 1
        videoseg = V(videoseg)
        mask = V(mask)
        optimizer.zero_grad()
        output1 = solver.forward(videoseg)
        print(output1)
        if torch.max(output1.data, 1).indices == mask:
            correct += 1
        loss1 = cost(output1, mask)
        loss = loss1
        # print('loss: ', loss)
        # print('output1, output2:', output1, output2)
        loss.backward()
        optimizer.step()
        loss1_sum += float(loss1)
        train_epoch_loss += float(loss)
        loss_log.append(loss.cpu().detach())
        print(jishu, loss)
        print("correct", correct)
    np.savetxt(savepath, loss_log, delimiter=',')
    loss1_sum /= len(data_loader_iter)
    train_epoch_loss /= len(data_loader_iter)
    precision = correct / len(data_loader_iter)
    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('alpha:', alpha, file=mylog)
    print('loss1:', loss1_sum, file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('precision:', precision, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)
    print('precision:', precision)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        torch.save(solver.state_dict(), 'weights/' + NAME + '.th')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    torch.save(solver.state_dict(), 'weights/' + NAME + str(epoch) + '.th')
    if no_optim > 3:
        if old_lr < 5e-9:
            break
        no_optim = 0
        solver.load_state_dict(torch.load('weights/' + NAME + '.th'))
        old_lr = update_lr(old_lr, 5, factor=True, mylog=mylog)
    mylog.flush()

print('Finish!', file=mylog)
print('Finish!')
mylog.close()
