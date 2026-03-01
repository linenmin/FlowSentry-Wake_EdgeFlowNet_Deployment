import tensorflow as tf
import sys
from functools import wraps
from misc.Decorators import *
from network.BaseLayers import *

class MultiScaleResNet(BaseLayers):

    def __init__(self, InputPH=None, Padding=None, NumOut=None, InitNeurons=None, ExpansionFactor=None, NumSubBlocks=None, NumBlocks=None, Suffix=None, UncType=None):
        super(MultiScaleResNet, self).__init__()
        if InputPH is None:
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        if InitNeurons is None:
            InitNeurons = 37
        if ExpansionFactor is None:
            ExpansionFactor = 2.0
        if NumSubBlocks is None:
            NumSubBlocks = 2
        if NumBlocks is None:
            NumBlocks = 1
        self.InitNeurons = InitNeurons
        self.ExpansionFactor = ExpansionFactor
        self.DropOutRate = 0.7
        self.NumSubBlocks = NumSubBlocks
        self.NumBlocks = NumBlocks
        if Suffix is None:
            Suffix = ''
        self.Suffix = Suffix
        if NumOut is None:
            NumOut = 1
        self.NumOut = NumOut
        self.currBlock = 0
        self.FeaturePyramid = None
        self.UncType = UncType
        if self.UncType == 'Aleatoric' or self.UncType == 'Inlier' or self.UncType == 'LinearSoftplus':
            self.NumOut *= 2
        self.kernel_size = (3, 3)
        self.strides = (2, 2)
        if Padding is None:
            Padding = 'same'
        self.padding = Padding

    @CountAndScope
    def OutputLayer(self, inputs=None, padding=None, rate=None, NumOut=None):
        if rate is None:
            rate = 0.5
        if NumOut is None:
            NumOut = self.NumOut
        flat = self.Flatten(inputs=inputs)
        drop = self.Dropout(inputs=flat, rate=rate)
        dense = self.Dense(inputs=drop, filters=NumOut, activation=None)
        return dense

    @CountAndScope
    def ResBlock(self, inputs=None, filters=None, kernel_size=None, strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        Net = self.ConvBNReLUBlock(inputs=inputs, filters=filters, padding=padding, strides=(1, 1))
        Net = self.Conv(inputs=Net, filters=filters, padding=padding, strides=(1, 1), activation=None)
        Net = self.BN(inputs=Net)
        Net = tf.add(Net, inputs)
        Net = self.ReLU(inputs=Net)
        return Net

    @CountAndScope
    def ResBlockTranspose(self, inputs=None, filters=None, kernel_size=None, strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        Net = self.ConvTransposeBNReLUBlock(inputs=inputs, filters=filters, padding=padding, strides=(1, 1))
        Net = self.ConvTranspose(inputs=Net, filters=filters, padding=padding, strides=(1, 1), activation=None)
        Net = self.BN(inputs=Net)
        Net = tf.add(Net, inputs)
        Net = self.ReLU(inputs=Net)
        return Net

    @CountAndScope
    def ResNetBlock(self, inputs):
        NumFilters = self.InitNeurons
        Net = self.ConvBNReLUBlock(inputs=inputs, filters=NumFilters, kernel_size=(7, 7))
        NumFilters = int(NumFilters * self.ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs=Net, filters=NumFilters, kernel_size=(5, 5))
        for count in range(self.NumSubBlocks):
            Net = self.ResBlock(inputs=Net, filters=NumFilters)
            NumFilters = int(NumFilters * self.ExpansionFactor)
            Net = self.Conv(inputs=Net, filters=NumFilters)
        Nets = []
        for count in range(self.NumSubBlocks):
            Net = self.ResBlockTranspose(inputs=Net, filters=NumFilters)
            NumFilters = int(NumFilters / self.ExpansionFactor)
            Net = self.ConvTranspose(inputs=Net, filters=NumFilters)
        feat_low = Net
        NetOut = self.ConvTranspose(inputs=Net, filters=self.NumOut, strides=(1, 1), kernel_size=(7, 7))
        Nets.append(NetOut)
        print(NetOut)
        NumFilters = int(NumFilters / self.ExpansionFactor)
        Net = self.ConvTransposeBNReLUBlock(inputs=Net, filters=NumFilters, kernel_size=(5, 5))
        feat_mid = Net
        NetOut = self.ConvTranspose(inputs=Net, filters=self.NumOut, strides=(1, 1), kernel_size=(7, 7))
        Nets.append(NetOut)
        print(NetOut)
        NumFilters = int(NumFilters / self.ExpansionFactor)
        Net = self.ConvTransposeBNReLUBlock(inputs=Net, filters=NumFilters, kernel_size=(7, 7))
        feat_high = Net
        Net = self.ConvTranspose(inputs=Net, filters=self.NumOut, kernel_size=(7, 7), strides=(1, 1), activation=None)
        Nets.append(Net)
        print(Net)
        self.FeaturePyramid = [feat_low, feat_mid, feat_high]
        return Nets

    def Network(self):
        OutNow = self.InputPH
        for count in range(self.NumBlocks):
            with tf.compat.v1.variable_scope('EncoderDecoderBlock' + str(count) + self.Suffix):
                OutNow = self.ResNetBlock(OutNow)
                self.currBlock += 1
        return OutNow

    def FeaturePyramidOutputs(self):
        if self.FeaturePyramid is None:
            self.Network()
        return self.FeaturePyramid