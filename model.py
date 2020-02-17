import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import Parameter
import math


class CRNN2D_elu2(nn.Module):
    def __init__(self, emb_size, classes_num):
        super(CRNN2D_elu2, self).__init__()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        
        self.Bn0 = nn.BatchNorm1d(128)

        self.Conv1 = nn.Conv2d(1, 64, (3,3), padding=(1,1))
        self.Bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.drop1 = nn.Dropout2d(p=0.1)

        self.Conv2 = nn.Conv2d(64, 128, (3,3), padding=(1,1))
        self.Bn2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d((4,2), stride=(4,2))
        self.drop2 = nn.Dropout2d(p=0.1)

        self.Conv3 = nn.Conv2d(128, 128, (3,3), padding=(1,1))
        self.Bn3 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d((4,2), stride=(4,2))
        self.drop3 = nn.Dropout2d(p=0.1)

        self.Conv4 = nn.Conv2d(128, 128, (3,3), padding=(1,1))
        self.Bn4 = nn.BatchNorm2d(128)
        self.mp4 = nn.MaxPool2d((4,2), stride=(4,2))
        self.drop4 = nn.Dropout2d(p=0.1)

        self.Conv1_1 = nn.Conv2d(64, 64, (3,3), padding=(1,1))
        self.Bn1_1 = nn.BatchNorm2d(64)
        self.mp1_1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.drop1_1 = nn.Dropout2d(p=0.1)

        self.Conv2_1 = nn.Conv2d(128, 128, (3,3), padding=(1,1))
        self.Bn2_1 = nn.BatchNorm2d(128)
        self.mp2_1 = nn.MaxPool2d((4,2), stride=(4,2))
        self.drop2_1 = nn.Dropout2d(p=0.1)

        self.Conv3_1 = nn.Conv2d(128, 128, (3,3), padding=(1,1))
        self.Bn3_1 = nn.BatchNorm2d(128)
        self.mp3_1 = nn.MaxPool2d((4,2), stride=(4,2))
        self.drop3_1 = nn.Dropout2d(p=0.1)

        self.Conv4_1 = nn.Conv2d(128, 128, (3,3), padding=(1,1))
        self.Bn4_1 = nn.BatchNorm2d(128)
        self.mp4_1 = nn.MaxPool2d((4,2), stride=(4,2))
        self.drop4_1 = nn.Dropout2d(p=0.1)

        self.gru1 = nn.GRU(128, 32, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(32, 32, num_layers=1, batch_first=True)
        self.drop5 = nn.Dropout(p=0.3)

        self.linear1 = nn.Linear(emb_size, classes_num)

    def forward(self, x, h):

        x = self.Bn0(x)

        x = x[:, None, :, :]

        # print(x.size())

        x = self.drop1(self.mp1(self.Bn1(self.elu(self.Conv1_1(self.Conv1(x))))))

        # print(x.size())

        x = self.drop2(self.mp2(self.Bn2(self.elu(self.Conv2_1(self.Conv2(x))))))

        # print(x.size())

        x = self.drop3(self.mp3(self.Bn3(self.elu(self.Conv3_1(self.Conv3(x))))))

        # print(x.size())

        x = self.drop4(self.mp4(self.Bn4(self.elu(self.Conv4_1(self.Conv4(x))))))

        # print(x.size())

        x = x.transpose(1, 3)

        x = torch.reshape(x, (x.size(0),x.size(1),-1))

        # print(x.size())

        x, h = self.gru1(x, h)

        # print(x.size())
        
        x, h = self.gru2(x, h)

        # print(x.size())

        x = self.drop5(x)

        x = torch.reshape(x, (x.size(0), -1))

        emb = x

        # print(x.size())

        x = self.softmax(self.linear1(x))

        # print(x.size())

        return x, emb



class CRNN2D_elu(nn.Module):
    def __init__(self, emb_size, classes_num):
        super(CRNN2D_elu, self).__init__()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        
        self.Bn0 = nn.BatchNorm1d(128)

        self.Conv1 = nn.Conv2d(1, 64, (3,3))
        self.Bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.drop1 = nn.Dropout2d(p=0.1)

        self.Conv2 = nn.Conv2d(64, 128, (3,3))
        self.Bn2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d((4,2), stride=(4,2))
        self.drop2 = nn.Dropout2d(p=0.1)

        self.Conv3 = nn.Conv2d(128, 128, (3,3))
        self.Bn3 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d((4,2), stride=(4,2))
        self.drop3 = nn.Dropout2d(p=0.1)

        self.Conv4 = nn.Conv2d(128, 128, (3,3))
        self.Bn4 = nn.BatchNorm2d(128)
        self.mp4 = nn.MaxPool2d((4,2), stride=(4,2))
        self.drop4 = nn.Dropout2d(p=0.1)

        self.gru1 = nn.GRU(128, 32, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(32, 32, num_layers=1, batch_first=True)
        self.drop5 = nn.Dropout(p=0.3)

        self.linear1 = nn.Linear(emb_size, classes_num)

    def forward(self, x, h):

        x = self.Bn0(x)

        x = x[:, None, :, :]

        x = self.drop1(self.mp1(self.Bn1(self.elu(self.Conv1(x)))))

        x = self.drop2(self.mp2(self.Bn2(self.elu(self.Conv2(x)))))

        x = self.drop3(self.mp3(self.Bn3(self.elu(self.Conv3(x)))))

        x = self.drop4(self.mp4(self.Bn4(self.elu(self.Conv4(x)))))

        x = x.transpose(1, 3)

        x = torch.reshape(x, (x.size(0),x.size(1),-1))

        x, h = self.gru1(x, h)
        
        x, h = self.gru2(x, h)

        x = self.drop5(x)

        x = torch.reshape(x, (x.size(0), -1))

        emb = x

        x = self.softmax(self.linear1(x))

        return x, emb

