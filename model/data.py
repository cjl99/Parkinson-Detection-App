import torch
import torch.utils.data as data
from torchvision import transforms

import os
from PIL import Image

def sumfigure(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #         transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]
    )
    img = Image.open(img_path)
    img = transform(img)
    return img

def default_loader(id, root,label):
    pics = list(filter(lambda x:x.find('.jpg')!=-1,os.listdir(root+id)))
#     pics = os.listdir(root+id)
    pics.sort()
    i = -1
    print(' ')
    print('videoseg:',id)
    for pic in pics:
        i += 2
        if i == 1:
            videoseg = sumfigure(root+id+'/'+pic)
        elif i ==3:
            videoseg = torch.cat((videoseg[None],sumfigure(root+id+'/'+pic)[None]),0)
        elif i<=64:
            videoseg = torch.cat((videoseg,sumfigure(root+id+'/'+pic)[None]),0)    

    label_id = id[4:8]
    videoname = list(filter(lambda x:x.find(str(label_id))!=-1,label['video']))
    index = label[label.video == videoname[0]].index.tolist()
    mask = label['label'][index[0]]

    return videoseg, mask

class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root,label):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root
        self.label = label

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root,self.label)
        img = torch.Tensor(img)
#         mask = torch.LongTensor([mask])
        return img, mask

    def __len__(self):
        return len(self.ids)