import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



class ALEXNET_nomax_pre(nn.Module):
    def __init__(self):
        super(ALEXNET_nomax_pre, self).__init__()
        self.pretrained_model = models.alexnet(pretrained=True).features
        self.max = nn.MaxPool2d(2,2)

    def forward(self, x1):
        if len(x1.shape) <4:
            x1 = torch.unsqueeze(x1, dim =0)
        x1 = self.pretrained_model(x1)
#    x1 = self.max(x1)
        return x1



class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.pretrained_model = models.vgg16(pretrained=True)
    

    def forward(self, x1):
        x1=x1.unsqueeze(0)
        x1 = self.pretrained_model(x1)
        return x1
