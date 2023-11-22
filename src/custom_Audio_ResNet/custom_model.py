# -*- coding: utf-8 -*-
# https://github.com/yikaiw/video-audio-resnet/blob/master/audio-resnet/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModelForAudioClassification

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool2d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(nn.Module):
    def __init__(self, block, layers, shortcut_type='B', num_classes=1000):
        # Configurables
        self._inplanes: int = 64
        self._init_outchannels: int = 64

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self._init_outchannels,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self._init_outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block=block, planes=self._init_outchannels, num_blocks=layers[0],
            shortcut_type=shortcut_type)
        self.layer2 = self._make_layer(
            block=block, planes=self._init_outchannels*2, num_blocks=layers[1],
            shortcut_type=shortcut_type,stride=2)
        self.layer3 = self._make_layer(
            block=block, planes=self._init_outchannels*4, num_blocks=layers[2],
            shortcut_type=shortcut_type,stride=2)
        self.layer4 = self._make_layer(
            block=block, planes=self._init_outchannels*8, num_blocks=layers[3],
            shortcut_type=shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1))
        self.fc = nn.Linear(
            in_features=self._init_outchannels*8 * block.expansion,
            out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self._inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self._inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self._inplanes, planes, stride, downsample))
        self._inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(planes * block.expansion, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x,dim=1)

class ResNetAudio_Config(PretrainedConfig):
    model_type = "resnet"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)

class ResNetForAudioClassification(PreTrainedModel):
    config_class = ResNetAudio_Config

    def __init__(self, config):
        super().__init__(config)
        resnet_arch = config.resnet_arch
        num_classes = config.num_labels

        assert resnet_arch in [
            "resnet_18", "resnet_34",
            "resnet_50", "resnet_101", "resnet_152"]

        if resnet_arch == "resnet_18":
            self._model = ResNet(
                block=BasicBlock,
                layers=[2, 2, 2, 2],
                shortcut_type='B',
                num_classes=num_classes)
        elif resnet_arch == "resnet_34":
            self._model = ResNet(
                block=BasicBlock,
                layers=[3, 4, 6, 3],
                shortcut_type='B',
                num_classes=num_classes)
        elif resnet_arch == "resnet_50":
            self._model = ResNet(
                block=Bottleneck,
                layers=[3, 4, 6, 3],
                shortcut_type='B',
                num_classes=num_classes)
        elif resnet_arch == "resnet_101":
            self._model = ResNet(
                block=Bottleneck,
                layers=[3, 4, 23, 3],
                shortcut_type='B',
                num_classes=num_classes)
        elif resnet_arch == "resnet_152":
            self._model = ResNet(
                block=Bottleneck,
                layers=[3, 8, 36, 3],
                shortcut_type='B',
                num_classes=num_classes)


    def forward(self, input_values, labels=None):
        logits = self._model(input_values)
        if labels is not None:
            loss = nn.functional.nll_loss(logits, labels)
            # loss = nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


if __name__ == "__main__":
    # Register
    ResNetAudio_Config.register_for_auto_class()
    ResNetForAudioClassification.register_for_auto_class("AutoModelForAudioClassification")

    AutoModelForAudioClassification.register(ResNetAudio_Config, ResNetForAudioClassification)

    # Config
    config = ResNetAudio_Config(
        resnet_arch="resnet_152",
        num_labels=4)
    # Create model
    resnet_model = ResNetForAudioClassification(config=config)
    resnet_model.load_state_dict(resnet_model.state_dict())
        # Load pretrained state_dict if needed

    # Login - terminal
        # huggingface-cli login
    # Login - notebook
        # from huggingface_hub import notebook_login
        # notebook_login()
    # Push to hub
    resnet_model.push_to_hub("AudioResNet")

    # Save model
    model = ResNetForAudioClassification.from_pretrained("dangkhoadl/AudioResNet")
    model.save_pretrained("src/custom_Audio_ResNet")
