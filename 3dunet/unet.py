import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, planes, stride=1):
        super(ConvBlock, self).__init__()
        in_planes, planes1, planes2 = planes
        self.conv1 = nn.Conv3d(in_planes, planes1, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm3d(planes1)
        self.conv2 = nn.Conv3d(planes1, planes2, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm3d(planes2)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class encoder(nn.Module):
    def __init__(self,in_planes):
        super(encoder, self).__init__()
        self.layer1=nn.Sequential(ConvBlock([in_planes, 32, 64]))
        self.p1=nn.MaxPool3d(kernel_size=2, stride=2)
        self.layer2=nn.Sequential(ConvBlock([64, 64, 128]))
        self.p2=nn.MaxPool3d(kernel_size=2, stride=2)
        self.layer3=nn.Sequential(ConvBlock([128, 128, 256]))
        self.p3=nn.MaxPool3d(kernel_size=2, stride=2)
        self.layer4=nn.Sequential(ConvBlock([256, 256, 512]))

    def forward(self, x):
        x = self.layer1(x)
        x = self.p1(x)
        x = self.layer2(x)
        x = self.p2(x)
        x = self.layer3(x)
        x = self.p3(x)
        x = self.layer4(x)
        return x

    def get_layers(self):
        return [self.layer1, self.layer2, self.layer3]


class decoder(nn.Module):
    def __init__(self,models):
        super(decoder, self).__init__()

        self.layer1=nn.Sequential(encoder(3),
                                  nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2,padding=1))
        self.layer2=nn.Sequential(ConvBlock([768, 256, 256]),
                                  nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2,padding=1))
        self.layer3=nn.Sequential(ConvBlock([384, 128, 128]),
                                  nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2,padding=1))
        self.layer4=nn.Sequential(ConvBlock([192, 64, 64]))
        self.conv1 = nn.Conv3d(64, 3, kernel_size=3, stride=1, bias=False)
        self.models = models

    def forward(self, x):
        x = self.layer1(x)
        x = torch.cat((self.models[2], x),1)
        x = self.layer2(x)
        x = torch.cat((self.models[1], x),1)
        x = self.layer3(x)
        x = torch.cat((self.models[0], x),1)
        x = self.layer4(x)
        x = self.conv1(x)
        x = F.softmax(x,dim=1)
        return x

class my3dunet(nn.Module):
    def __init__(self,in_planes):
        super(my3dunet,self).__init__()
        self.en = encoder(in_planes)
        self.de = decoder(self.en.get_layers())
    def forward(self, x):
        x = self.en(x)
        x = self.de(x)
        return x



