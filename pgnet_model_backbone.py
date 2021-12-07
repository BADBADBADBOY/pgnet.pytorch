# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# __all__ = ["ResNet"]


# class ConvBNLayer(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=1,
#             groups=1,
#             is_vd_mode=False,
#             act=None,
#             name=None, ):
#         super(ConvBNLayer, self).__init__()

#         self.is_vd_mode = is_vd_mode
#         self._pool2d_avg = nn.AvgPool2d(
#             kernel_size=2, stride=2, padding=0, ceil_mode=True)
#         self._conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=(kernel_size - 1) // 2,
#             groups=groups)

#         self._batch_norm = nn.BatchNorm2d(out_channels)
#         self.if_act = False
#         if act is not None:
#             self.relu = nn.ReLU()
#             self.if_act = True

#     def forward(self, inputs):
#         y = self._conv(inputs)
#         y = self._batch_norm(y)
#         if self.if_act:
#             y = self.relu(y)
#         return y


# class BottleneckBlock(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  stride,
#                  shortcut=True,
#                  if_first=False,
#                  name=None):
#         super(BottleneckBlock, self).__init__()

#         self.conv0 = ConvBNLayer(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             act='relu',
#             name=name + "_branch2a")
#         self.conv1 = ConvBNLayer(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             stride=stride,
#             act='relu',
#             name=name + "_branch2b")
#         self.conv2 = ConvBNLayer(
#             in_channels=out_channels,
#             out_channels=out_channels * 4,
#             kernel_size=1,
#             act=None,
#             name=name + "_branch2c")

#         if not shortcut:
#             self.short = ConvBNLayer(
#                 in_channels=in_channels,
#                 out_channels=out_channels * 4,
#                 kernel_size=1,
#                 stride=stride,
#                 is_vd_mode=False if if_first else True,
#                 name=name + "_branch1")

#         self.shortcut = shortcut

#     def forward(self, inputs):
#         y = self.conv0(inputs)
#         conv1 = self.conv1(y)
#         conv2 = self.conv2(conv1)

#         if self.shortcut:
#             short = inputs
#         else:
#             short = self.short(inputs)
#         y = torch.add(short, conv2)
#         y = F.relu(y)
#         return y


# class BasicBlock(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  stride,
#                  shortcut=True,
#                  if_first=False,
#                  name=None):
#         super(BasicBlock, self).__init__()
#         self.stride = stride
#         self.conv0 = ConvBNLayer(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             stride=stride,
#             act='relu',
#             name=name + "_branch2a")
#         self.conv1 = ConvBNLayer(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             act=None,
#             name=name + "_branch2b")

#         if not shortcut:
#             self.short = ConvBNLayer(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=1,
#                 stride=1,
#                 is_vd_mode=False if if_first else True,
#                 name=name + "_branch1")

#         self.shortcut = shortcut

#     def forward(self, inputs):
#         y = self.conv0(inputs)
#         conv1 = self.conv1(y)

#         if self.shortcut:
#             short = inputs
#         else:
#             short = self.short(inputs)
#         y = torch.add(short, conv1)
#         y = F.relu(y)
#         return y


# class ResNet(nn.Module):
#     def __init__(self, in_channels=3, layers=50, **kwargs):
#         super(ResNet, self).__init__()

#         self.layers = layers
#         supported_layers = [18, 34, 50, 101, 152, 200]
#         assert layers in supported_layers, \
#             "supported layers are {} but input layer is {}".format(
#                 supported_layers, layers)

#         if layers == 18:
#             depth = [2, 2, 2, 2]
#         elif layers == 34 or layers == 50:
#             # depth = [3, 4, 6, 3]
#             depth = [3, 4, 6, 3, 3]
#         elif layers == 101:
#             depth = [3, 4, 23, 3]
#         elif layers == 152:
#             depth = [3, 8, 36, 3]
#         elif layers == 200:
#             depth = [3, 12, 48, 3]
#         num_channels = [64, 256, 512, 1024,
#                         2048] if layers >= 50 else [64, 64, 128, 256]
#         num_filters = [64, 128, 256, 512, 512]

#         self.conv1_1 = ConvBNLayer(
#             in_channels=in_channels,
#             out_channels=64,
#             kernel_size=7,
#             stride=2,
#             act='relu',
#             name="conv1_1")
#         self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.stages = []
#         self.out_channels = [3, 64]
#         # num_filters = [64, 128, 256, 512, 512]
#         if layers >= 50:
#             for block in range(len(depth)):
#                 block_list = []
#                 shortcut = False
#                 for i in range(depth[block]):
#                     if layers in [101, 152] and block == 2:
#                         if i == 0:
#                             conv_name = "res" + str(block + 2) + "a"
#                         else:
#                             conv_name = "res" + str(block + 2) + "b" + str(i)
#                     else:
#                         conv_name = "res" + str(block + 2) + chr(97 + i)
#                     bottleneck_block = BottleneckBlock(
#                             in_channels=num_channels[block]
#                             if i == 0 else num_filters[block] * 4,
#                             out_channels=num_filters[block],
#                             stride=2 if i == 0 and block != 0 else 1,
#                             shortcut=shortcut,
#                             if_first=block == i == 0,
#                             name=conv_name)
#                     shortcut = True
#                     block_list.append(bottleneck_block)
#                 self.out_channels.append(num_filters[block] * 4)
#                 self.stages.append(nn.Sequential(*block_list))
#             self.stages = nn.Sequential(*self.stages)
#         else:
#             for block in range(len(depth)):
#                 block_list = []
#                 shortcut = False
#                 for i in range(depth[block]):
#                     conv_name = "res" + str(block + 2) + chr(97 + i)
#                     basic_block = BasicBlock(
#                             in_channels=num_channels[block]
#                             if i == 0 else num_filters[block],
#                             out_channels=num_filters[block],
#                             stride=2 if i == 0 and block != 0 else 1,
#                             shortcut=shortcut,
#                             if_first=block == i == 0,
#                             name=conv_name)
#                     shortcut = True
#                     block_list.append(basic_block)
#                 self.out_channels.append(num_filters[block])
#                 self.stages.append(nn.Sequential(*block_list))
#             self.stages = nn.Sequential(*self.stages)
#     def forward(self, inputs):
#         out = [inputs]
#         y = self.conv1_1(inputs)
#         out.append(y)
#         y = self.pool2d_max(y)
#         for block in self.stages:
#             y = block(y)
#             out.append(y)
#         return out


#-*- coding:utf-8 _*-
"""
@author:fxw
@file: det_resnet.py.py
@time: 2020/08/07
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def load_pre_model(model,pre_model_path):
    pre_dict = torch.load(pre_model_path)
    model_pre_dict = {}
    for key in model.state_dict().keys():
        if('model.module.backbone.'+key in pre_dict.keys()):
            model_pre_dict[key] = pre_dict['model.module.backbone.'+key]
        else:
            model_pre_dict[key] = model.state_dict()[key]
    model.load_state_dict(model_pre_dict)
    return model

def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        # self.conv2 = conv3x3(planes, planes)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   padding=1, bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                from models.dcn import DeformConv
                conv_op = DeformConv
                offset_channels = 18
            else:
                from models.dcn import ModulatedDeformConv
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                padding=1)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                deformable_groups=deformable_groups,
                bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                from models.dcn import DeformConv
                conv_op = DeformConv
                offset_channels = 18
            else:
                from models.dcn import ModulatedDeformConv
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes, deformable_groups * offset_channels,
                kernel_size=3,
                padding=1)
            self.conv2 = conv_op(
                planes, planes, kernel_size=3, padding=1, stride=stride,
                deformable_groups=deformable_groups, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 dcn=None, stage_with_dcn=(False, False, False, False)):
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dcn=dcn)
        self.layer5 = self._make_layer(
            block, 512, layers[4], stride=2, dcn=dcn)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.smooth = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x.clone()
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)

        return [x0,x1,x2,x3,x4,x5,x6]


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet18']), strict=False)
    return model


def deformable_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                   dcn=dict(modulated=True,
                            deformable_groups=1,
                            fallback_on_stride=False),
                   stage_with_dcn=[False, True, True, True], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False,load_url=False,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3,3], **kwargs)
    if pretrained:
        if load_url:
            model.load_state_dict(model_zoo.load_url(
                model_urls['resnet50']), strict=False)
        else:
            model = load_pre_model(model,'./pre_model/pre-trained-model-synthtext-resnet50.pth')
    return model


def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3,3],
                   dcn=dict(modulated=True,
                            deformable_groups=1,
                            fallback_on_stride=False),
                   stage_with_dcn=[False, True, True, True],
                   **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet152']), strict=False)
    return model


