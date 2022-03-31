import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MyMobileNetv3(nn.Module):
    def __init__(self):
        super(MyMobileNetv3, self).__init__()
        self.layer1 = nn.Sequential(mobilenet_v3_small(num_classes=2))
        self.layer2 = nn.Sequential(nn.Flatten(start_dim=0))
        self.layer3 = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 2))
        self.layer4 = nn.Sequential(nn.Softmax(dim=0))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out).reshape(1, 2)
        return out

    def forward2(self, x):
        return self.forward(x)
