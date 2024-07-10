import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from Xception_model import xception
from Xception_model_correct import xception as xception2


img_shape = (3, 299, 299)
cuda = True if torch.cuda.is_available() else False


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, padding=None):
        super(conv_block, self).__init__()

        if padding is not None:
            padding = padding
        else:
            padding = 1

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=padding, bias=True, ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, padding=None, upsample_size=None):
        super(up_conv, self).__init__()
        if padding is not None:
            padding = padding
        else:
            padding = 1

        if upsample_size is not None:
            upsample = nn.Upsample(size=upsample_size)
        else:
            upsample = nn.Upsample(scale_factor=2)
        self.up = nn.Sequential(
            upsample,
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class DeepfakeDetector(nn.Module):

    def __init__(self, pretrained=True, finetuning=False, architecture='Xception', frozen_params=50, output_features=False):
        super(DeepfakeDetector, self).__init__()

        self.output_features = output_features

        if architecture == 'Xception':
            self.model = xception2(pretrained=pretrained)
            self.model.fc = nn.Sequential()
            self.fc_xc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                       nn.Dropout(p=0.25), nn.Linear(256, 1), nn.Sigmoid())
            # self.fc_xc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
            #                            nn.Dropout(p=0), nn.Linear(256, 1), nn.Sigmoid())

        if finetuning:
            # 154 parameters for XceptionNET
            i = 0
            for param in self.model.parameters():
                i += 0
                if i < frozen_params:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def forward(self, x):

        x = self.model(x)
        out = self.fc_xc(x)

        if self.output_features:
            return x
        else:
            return out


class CNN(nn.Module):

    def __init__(self, pretrained=True, finetuning=False, architecture='Xception', frozen_params=50, output_features=False, use_bn=True):
        super(CNN, self).__init__()

        self.output_features = output_features

        if architecture == 'Xception':
            self.model = xception(pretrained=pretrained, use_bn=use_bn)
            self.model.fc = nn.Sequential()
            self.fc_xc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                       nn.Dropout(p=0.25), nn.Linear(256, 1), nn.Sigmoid())

        if finetuning:
            # 154 parameters for XceptionNET
            i = 0
            for param in self.model.parameters():
                i += 0
                if i < frozen_params:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def forward(self, x):

        x = self.model(x)
        out = self.fc_xc(x)

        if self.output_features:
            return x
        else:
            return out



class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=3):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Up5 = up_conv(filters[4], filters[3], padding='same', upsample_size=(37, 37))
        self.Up3 = up_conv(filters[2], filters[1], upsample_size=(149, 149))
        self.Up2 = up_conv(filters[1], filters[0], upsample_size=(299, 299))

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out

if __name__ == "__main__":

    random_tensor = torch.rand(1, 3, 299, 299)
    g = AttU_Net()

    a=g(random_tensor)
    print(1)

