import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

def ConvBlock(planes):
    in_planes, planes1, planes2 = planes
    return nn.Sequential(nn.Conv3d(in_planes, planes1, kernel_size=3, padding=1),
                         nn.BatchNorm3d(planes1),
                         nn.ReLU(),
                         nn.Conv3d(planes1, planes2, kernel_size=3, padding=1),
                         nn.BatchNorm3d(planes2),
                         nn.ReLU())


class my3dunet(nn.Module):
    def __init__(self,in_planes, out_planes):
        super(my3dunet,self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.layer1=ConvBlock([self.in_planes, 32, 64])
        self.d1=nn.MaxPool3d(kernel_size=2, stride=2)
        self.layer2=ConvBlock([64, 64, 128])
        self.d2=nn.MaxPool3d(kernel_size=2, stride=2)
        self.layer3=ConvBlock([128, 128, 256])
        self.d3=nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer4=ConvBlock([256, 256, 512])

        self.u4=nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.layer5=ConvBlock([768, 256, 256])
        self.u3=nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.layer6=ConvBlock([384, 128, 128])
        self.u2=nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.layer7=ConvBlock([192, 64, 64])

        self.conv = nn.Conv3d(64, self.out_planes, kernel_size=3, stride=1, padding=1)
        self.s = nn.Softmax(dim=1)

    def forward(self, x):
        l1 = self.layer1(x)
        # print("l1 size: {}".format(l1.size()))
        down1 = self.d1(l1)
        # print("down1 size: {}".format(down1.size()))
        l2 = self.layer2(down1)
        # print("l2 size: {}".format(l2.size()))
        down2 = self.d2(l2)
        # print("down2 size: {}".format(down2.size()))
        l3 = self.layer3(down2)
        # print("l3 size: {}".format(l3.size()))
        down3 = self.d3(l3)
        # print("down3 size: {}".format(down3.size()))

        l4 = self.layer4(down3)
        # print("l4 size: {}".format(l4.size()))

        up4 = self.u4(l4)
        # print("up4 size: {}".format(up4.size()))
        c1 = torch.cat((l3, up4),1)
        # print("c1 size: {}".format(c1.size()))
        l5 = self.layer5(c1)
        # print("l5 size: {}".format(l5.size()))

        up3 = self.u3(l5)
        # print("up3 size: {}".format(up3.size()))
        c2 = torch.cat((l2, up3),1)
        # print("c2 size: {}".format(c2.size()))
        l6 = self.layer6(c2)
        # print("l6 size: {}".format(l6.size()))

        up2 = self.u2(l6)
        # print("up2 size: {}".format(up2.size()))
        c3 = torch.cat((l1, up2),1)
        # print("c3 size: {}".format(c3.size()))
        l7 = self.layer7(c3)
        # print("l7 size: {}".format(l7.size()))
        out = self.conv(l7)
        out = self.s(out)

        return out

'''device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = torch.Tensor(1, 3, 333, 256, 256)
x.to(device)
print("x size: {}".format(x.size()))
  
model = my3dunet(in_planes=3, out_planes=3) 
out = model(x)
print("out size: {}".format(out.size()))

net=my3dunet(3,3)
inputs=torch.randn(1,3,256,512,512)
flops, params = profile(net, (inputs,))
print('flops: ', flops, 'params: ',params)'''
