### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
from torch.functional import Tensor

"""This script defines the network.
"""

class MyNetwork(nn.Module):
    '''
	Args:
        inputs: A Tensor representing a batch of input images.
        training: A boolean. Used by operations that work differently
            in training and testing phases such as batch normalization.
    Return:
        The output Tensor of the network.
    '''
    def __init__(self, configs):
        super(MyNetwork, self).__init__()
        self.configs = configs
        self.stages = [64, 64 * configs['widen_factor'], 128 * configs['widen_factor'], 256 * configs['widen_factor']]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = stack_layer(configs, self.stages[0], self.stages[1], 1)
        self.stage_2 = stack_layer(configs, self.stages[1], self.stages[2], 2)
        self.stage_3 = stack_layer(configs, self.stages[2], self.stages[3], 2)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=8, stride=1)
        self.classifier = nn.Linear(self.stages[3], configs['num_classes'])
        nn.init.kaiming_normal_(self.classifier.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
                
    def forward(self, input):
        output = self.conv_1_3x3(input)
        output = self.bn_1(output)
        output = self.relu(output)
        output = self.stage_1(output)
        output = self.stage_2(output)
        output = self.stage_3(output)
        output = self.avg_pool2d(output)
        output = output.view(-1, self.stages[3])
        output = self.classifier(output)
        return output


#############################################################################
# Blocks building the network
#############################################################################


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck:
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
    """
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, input):
        output = self.conv_reduce(input)
        output = self.bn_reduce(output)
        output = self.relu(output)
        output = self.conv_conv(output)
        output = self.relu(output)
        output = self.conv_expand(output)
        output = self.bn_expand(output)
        output += self.shortcut(input)
        output = self.relu(output)
        return output




class stack_layer(nn.Module):
    """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
    """
    def __init__(self, configs, in_channels, out_channels, pool_stride=2):
        super(stack_layer, self).__init__()
        self.configs = configs
        self.blocks = nn.ModuleList()
        for bottleneck in range(configs['resnet_size']):
            if bottleneck == 0:
                self.blocks.append(ResNeXtBottleneck(in_channels, out_channels, pool_stride, 
                        configs['cardinality'], configs['base_width'], configs['widen_factor']))
            else:
                self.blocks.append(ResNeXtBottleneck(out_channels, out_channels, 1, 
                        configs['cardinality'], configs['base_width'], configs['widen_factor']))

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        for i in range(self.configs['resnet_size']):
            outputs = self.blocks[i](outputs)
        return outputs          
       


### END CODE HERE