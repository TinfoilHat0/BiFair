import torch
import torch.nn as nn
from torchvision import models as pt_models


def get_model(data, input_size=None):
    if data == 'celebA':
        model = pt_models.resnet18(pretrained=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, 1, bias=True)
           
    # elif data == 'adult':
    #     model = FFN(input_size=input_size) 
    else:
        model = LogisticReg(input_size=input_size)
  
    return model


class FFN(torch.nn.Module):

    def __init__(self, input_size, p_dropout=0.5, n_class=1):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, n_class)
    
        self.dropout = nn.Dropout(p=p_dropout)
        
    def forward(self, x):
        x = self.dropout(self.fc1(x).relu())
        x = self.fc2(x)
        return x


class LogisticReg(nn.Module):
    def __init__(self, input_size, n_class=1):
        super(LogisticReg, self).__init__()
        self.linear = nn.Linear(input_size, n_class)
        
    def forward(self, x):
        x = self.linear(x)
        return x
    



