# 添加精度，召回率，F1指标的计算
import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from model.data import sumfigure
from model.MyMoblenet2 import MyMobileNetv2

def predictor(videoseg, net, use_gpu):
    x = Variable(videoseg, requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    out = net(x)
    predicted = out.cpu().data.numpy()
    #     predicted = torch.max(out.data, 1).indices
    return predicted


def sum_data(sourcepath, picfile):
    pics = os.listdir(sourcepath + picfile)
    pics.sort()
    i = -1
    for pic in pics:
        i += 2
        if i == 1:
            videoseg = sumfigure(sourcepath + picfile + '/' + pic)
        elif i == 3:
            videoseg = torch.cat((videoseg[None], sumfigure(sourcepath + picfile + '/' + pic)[None]), 0)
        elif i <= 64:
            videoseg = torch.cat((videoseg, sumfigure(sourcepath + picfile + '/' + pic)[None]), 0)
    return videoseg


def mainfunction(model_s, sourcepath_spatial):
    filedir = os.listdir(sourcepath_spatial)
    correct = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for picfile in filedir:
        judge = 0
        if int(picfile[4:8]) < 800:
            judge = 1
        videoseg_s = sum_data(sourcepath_spatial, picfile)
        pre_spatical = predictor(videoseg_s, model_s, False)
        pre = pre_spatical
        pre = np.argmax(pre)
        if judge == pre:
            correct += 1
            if judge == 1:
                TP += 1
            else:
                TN += 1
        elif judge == 1:
            FN += 1
        else:
            FP += 1
        print('judge:', judge, 'pre', pre, 'correct', correct, 'TP,FP,FN,TN', TP, FP, FN, TN, pre_spatical)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    specificity = TN / (TN + FP)
    accuracy = correct / len(filedir)
    return precision, recall, F1, accuracy, specificity


def predict(modelpath_spatial, model_spaticial, sourcepath_spatial):
    # model_s = generate_model(model_depth=101).cuda()
    model_s = MyMobileNetv2()
    model_s.load_state_dict(torch.load(modelpath_spatial + model_spaticial, map_location=torch.device('cuda')))
    # model_t = generate_model(model_depth=101).cuda()
    precision, recall, F1, accuracy, specificity = mainfunction(model_s,  sourcepath_spatial)
    return precision, recall, F1, accuracy, specificity




sourcepath_spatial = './dataset/crop_face/Testset/'  # 空间流输入数据

modelpath_spatial = './weights/'  # 空间流模型位置
model_spaticial = 'spacial_model20.th'

# mylog = open('./test_logs/test.log', 'a')

precision, recall, F1, accuracy, specificity = predict(modelpath_spatial, model_spaticial,
                                                     sourcepath_spatial)
print('**********')
print('model:', model_spaticial)
print('precision:', precision)
print('recall:', recall)
print('F1:', F1)
print('accuracy:', accuracy)
print('model:', model_spaticial, ':precision:', precision, ':recall:', recall, ':F1:', F1, ':accuracy:',
      accuracy, ':specificity:', specificity)
#     print('precision:',precision, file = mylog)
#     print('recall:',recall, file = mylog)
#     print('F1:',F1, file = mylog)
#     print('accuracy:',accuracy, file = mylog)

