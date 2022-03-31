import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class MyMobileNetv2(nn.Module):
    def __init__(self):
        super(MyMobileNetv2, self).__init__()
        self.layer1 = nn.Sequential(mobilenet_v2(num_classes=2))
        self.layer2 = nn.Sequential(nn.Flatten(start_dim=0), nn.Linear(64, 64))
        self.layer3 = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16), nn.Linear(16, 8), nn.Linear(8, 2))
        self.layer4 = nn.Sequential(nn.Softmax(dim=0))

    def forward(self, x):
        arr = self.layer1(x[:, :, 0, :, :])
        for i in range(1, 32):
            out = self.layer1(x[:, :, i, :, :])
            arr = torch.cat((arr, out))

        out = self.layer2(arr)
        out = self.layer3(out)
        out = self.layer4(out).reshape(1, 2)
        return out

    def forward2(self, x):
        return self.forward(x)
