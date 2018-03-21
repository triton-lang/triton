import torch
import torch.nn as nn
import isaac.pytorch as sc
import torch.utils.model_zoo as model_zoo

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_num, out_num, stride = 1, downsample = None, dim=2):
        super(BasicBlock, self).__init__()
        self.conv1 =  sc.ConvType[dim](in_num, out_num, 3, bias = True, activation = 'relu', alpha = 0, padding=1, stride=stride)
        self.conv2 =  sc.ConvType[dim](out_num, out_num, 3, bias = True, activation = 'relu', alpha = 0, residual = 'add', padding=1)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out, residual)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_num, out_num, stride=1, downsample=None, dim=2):
        super(Bottleneck, self).__init__()
        self.conv1 = sc.ConvType[dim](in_num, out_num, kernel_size=1, bias=True, activation='relu')
        self.conv2 = sc.ConvType[dim](out_num, out_num, kernel_size=3, stride=stride, padding=1, bias=True, activation='relu')
        self.conv3 = sc.ConvType[dim](out_num, out_num*4, kernel_size=1, bias=True, activation='relu', residual='add')
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out, residual)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_num = 64
        self.conv1 = sc.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True, activation='relu', alpha=0)
        self.maxpool = sc.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = sc.AvgPool2d(kernel_size=7, stride=1)
        self.fc = sc.Linear(512 * block.expansion, num_classes)
        # Define first and last layer (for conversion to/from int8x4)
        self.conv1.is_first = True
        self.fc.is_last = True
        # Use bias initialized to zero instead of no-bias because batch-norm will be folded
        for x in self.modules():
            if isinstance(x, sc.ConvNd):
                x.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        # Downsampling
        downsample = None
        if stride != 1 or self.in_num != planes * block.expansion:
            downsample = sc.Conv2d(self.in_num, planes*block.expansion, kernel_size=1, stride=stride, bias=True)
        # Layers
        layers = [block(self.in_num, planes, stride, downsample)]
        self.in_num = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_num, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet(name, **kwargs):
    pretrained_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    }

    blocks = {
        'resnet18': BasicBlock,
        'resnet34': BasicBlock,
        'resnet50': Bottleneck,
        'resnet101': Bottleneck,
        'resnet152': Bottleneck
    }

    layers = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3],
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3]
    }

    model = ResNet(blocks[name], layers[name], **kwargs).cuda()
    sc.convert(model, model_zoo.load_url(pretrained_urls[name]))
    return model
