import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps
import misc.TFUtils as tu
from misc.Decorators import *

class BaseLayers(object):

    def __init__(self):
        self.CurrBlock = 0

    @CountAndScope
    def ConvBNReLUBlock(self, inputs=None, filters=None, kernel_size=None, strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        conv = self.Conv(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        bn = self.BN(conv)
        Output = self.ReLU(bn)
        return Output

    @CountAndScope
    def SeparableConvBNReLUBlock(self, inputs=None, filters=None, kernel_size=None, strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        conv = self.SeparableConv(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        bn = self.BN(conv)
        Output = self.ReLU(bn)
        return Output

    @CountAndScope
    def ConvTransposeBNReLUBlock(self, inputs=None, filters=None, kernel_size=None, strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        conv = self.ConvTranspose(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        bn = self.BN(conv)
        Output = self.ReLU(bn)
        return Output

    @CountAndScope
    def SeparableConv(self, inputs=None, filters=None, kernel_size=None, strides=None, padding=None, activation=None, name=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        Output = tf.compat.v1.layers.separable_conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, name=name)
        return Output

    @CountAndScope
    def Conv(self, inputs=None, filters=None, kernel_size=None, strides=None, padding=None, activation=None, name=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        Output = tf.compat.v1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, name=name)
        return Output

    @CountAndScope
    def ConvTranspose(self, inputs=None, filters=None, kernel_size=None, strides=None, padding=None, activation=None, name=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        Output = tf.compat.v1.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, name=name)
        return Output

    @CountAndScope
    def BN(self, inputs=None):
        Output = tf.compat.v1.layers.batch_normalization(inputs=inputs)
        return Output

    @CountAndScope
    def ReLU(self, inputs=None):
        Output = tf.compat.v1.nn.relu(inputs)
        return Output

    @CountAndScope
    def Concat(self, inputs=None, axis=0):
        Output = tf.compat.v1.concat(values=inputs, axis=axis)
        return Output

    @CountAndScope
    def Flatten(self, inputs=None):
        Shape = inputs.get_shape().as_list()
        Dim = np.prod(Shape[1:])
        Output = tf.reshape(inputs, [-1, Dim])
        return Output

    @CountAndScope
    def Dropout(self, inputs=None, rate=None):
        if rate is None:
            rate = 0.5
        Output = tf.compat.v1.layers.dropout(inputs, rate=rate)
        return Output

    @CountAndScope
    def Dense(self, inputs=None, filters=None, activation=None, name=None):
        Output = tf.compat.v1.layers.dense(inputs, units=filters, activation=activation, name=name)
        return Output