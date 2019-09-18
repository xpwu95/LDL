import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.nn.functional as F
import numpy as np


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=0.001)
        if m.bias is not None:
            init.constant(m.bias, 0)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        # self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.AdaptiveMaxPool2d(1))
        # self.fc.apply(weights_init)
        #

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512 * block.expansion, 4)
        # self.fc = nn.Sequential(nn.Linear(2048, 512),
        #                         nn.ReLU(inplace=True),
        #                         nn.Dropout(p=0.5),
        #                         nn.Linear(512, 4))
        self.fc.apply(weights_init)

        self.counting = nn.Linear(512 * block.expansion, 65)
        # self.counting = nn.Sequential(nn.Linear(2048, 512),
        #                               nn.ReLU(inplace=True),
        #                               nn.Dropout(p=0.5),
        #                               nn.Linear(512, 65))
        self.counting.apply(weights_init)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.bn1.apply(set_bn_fix)
        self.layer1.apply(set_bn_fix)
        self.layer2.apply(set_bn_fix)
        self.layer3.apply(set_bn_fix)
        self.layer4.apply(set_bn_fix)

        # self.conv2 = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=True),
        #                            nn.ReLU(inplace=True))
        # self.conv2.apply(weights_init)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, x_new):

        batch_size = len(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # if batch_size % 16 == 0 and self.training == True:
        #     x_new = F.adaptive_max_pool2d(x[x_new].view(len(x_new), 64, 112, 112), 56)
        #     x = torch.cat((x, x_new))

        # if batch_size % 16 == 0 and self.training == True:
        #     for i in range(len(x_new)):
        #         if i == 0:
        #             temp = x[x_new[0]].view(1, 4, 64, 56, 56)
        #         else:
        #             temp = torch.cat((temp, x[x_new[i]].view(1, 4, 64, 56, 56)), 0)
        #     temp = temp.view(len(x_new), 64, 112, 112)
        #     x_new = F.adaptive_max_pool2d(temp, 56)
            # x_new = self.conv2(temp)
            # x = torch.cat((x, x_new))
            # print('te')

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        '''
        if batch_size % 16 == 0 and self.training == True:
            for i in range(batch_size):
                if i == 0:
                    temp = x[x_new[0]].view(1, 4, 2048, 7, 7)
                else:
                    temp = torch.cat((temp, x[x_new[i]].view(1, 4, 2048, 7, 7)), 0)
            temp = temp.view(batch_size, 2048, 14, 14)
            x_new = F.adaptive_max_pool2d(temp, 7)
            # x_new = self.conv2(temp)
            x = torch.cat((x, x_new))
            # print('te')
        '''
        # if batch_size % 16 == 0 and self.training == True:
        #     x_new = F.adaptive_max_pool2d(x[x_new].view(len(x_new), 2048, 14, 14), 7)
        #     x = torch.cat((x, x_new))

        # x = F.adaptive_max_pool2d(x, 4)  # .....

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        cls = self.fc(x)
        cou = self.counting(x)

        cls = F.softmax(cls) + 1e-4
        cou = F.softmax(cou) + 1e-4

        cou2cls = torch.stack((torch.sum(cou[:, :5], 1), torch.sum(cou[:, 5:20], 1), torch.sum(cou[:, 20:50], 1),
                               torch.sum(cou[:, 50:], 1)), 1)
        # cou2cls = torch.log(cou2cls)

        # Exception
        # cou2cou = torch.sum(cou * torch.from_numpy(np.array(range(1, 66))).float().cuda(), 1)

        return cls, cou, cou2cls

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            # self.conv1.eval()
            # self.bn1.eval()
            # self.relu.eval()
            # self.maxpool.eval()
            # self.layer1.eval()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.bn1.apply(set_bn_eval)
            self.layer1.apply(set_bn_eval)
            self.layer2.apply(set_bn_eval)
            self.layer3.apply(set_bn_eval)
            self.layer4.apply(set_bn_eval)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
