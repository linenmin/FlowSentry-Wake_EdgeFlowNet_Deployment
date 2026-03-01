import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
import sys
import os
import glob
import re
import misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import misc.TFUtils as tu
from misc.DataHandling import *
from misc.BatchCreationTF import *
from misc.Decorators import *
from misc.FlowVisUtilsTF import *
import importlib
from datetime import datetime
import getpass
import copy
import platform
import pdb
import misc.FlowPolar as fp
import misc.FlowShift as fs

def AssemblePreds(I):
    tf_u = tf.concat((I[:, :, :, 0:2], I[:, :, :, 2:4]), axis=2)
    tf_b = tf.concat((I[:, :, :, 4:6], I[:, :, :, 6:8]), axis=2)
    tf_p = tf.concat((tf_u, tf_b), axis=1)
    return tf_p

def AssembleImages(I):
    tf_u = tf.concat((I[:, :, :, 0:3], I[:, :, :, 3:6]), axis=2)
    tf_b = tf.concat((I[:, :, :, 6:9], I[:, :, :, 9:12]), axis=2)
    tf_p = tf.concat((tf_u, tf_b), axis=1)
    return tf_p

def SSIM(I1, I2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = tf.nn.avg_pool(I1, ksize=(3, 3), strides=(1, 1), padding='SAME')
    mu_y = tf.nn.avg_pool(I2, ksize=(3, 3), strides=(1, 1), padding='SAME')
    sigma_x = tf.nn.avg_pool(I1 ** 2, ksize=(3, 3), strides=(1, 1), padding='SAME') - mu_x ** 2
    sigma_y = tf.nn.avg_pool(I2 ** 2, ksize=(3, 3), strides=(1, 1), padding='SAME') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool(I1 * I2, ksize=(3, 3), strides=(1, 1), padding='SAME') - mu_x * mu_y
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    return SSIM

def LossMultiscaleUncertainity(Label1PH, prVal, Args):
    prValAccum = None
    prLogVarAccum = None
    loss_scales = [0.125, 0.25, 0.5, 1]
    Eps = 0.001
    for prVali, loss_scale in zip(prVal, loss_scales):
        if prValAccum == None:
            prValAccum = prVali[..., :Args.NumOut]
            Labeli = tf.compat.v1.image.resize(Label1PH, [prValAccum.shape[1], prValAccum.shape[2]])
            lossOpticali = tf.reduce_mean(tf.abs(prValAccum - Labeli))
            lossOpticalAccum = loss_scale * lossOpticali
            prLogVarAccum = prVali[..., Args.NumOut:]
            lossUnc = tf.reduce_mean(1 / tf.maximum(tf.math.softplus(prLogVarAccum + Eps), Eps) * tf.abs(prValAccum - Labeli)) + tf.reduce_mean(tf.math.softplus(prLogVarAccum))
            lossUncAccum = loss_scale * lossUnc
            continue
        prValAccum = tf.compat.v1.image.resize(prValAccum, [prVali.shape[1], prVali.shape[2]])
        prValAccum += prVali[..., :Args.NumOut]
        Labeli = tf.compat.v1.image.resize(Label1PH, [prValAccum.shape[1], prValAccum.shape[2]])
        lossOpticali = tf.reduce_mean(tf.abs(prValAccum - Labeli))
        lossOpticalAccum += loss_scale * lossOpticali
        prLogVarAccum = tf.compat.v1.image.resize(prLogVarAccum, [prVali.shape[1], prVali.shape[2]])
        prLogVarAccum += prVali[..., Args.NumOut:]
        lossUnc = tf.reduce_mean(1 / tf.maximum(tf.math.softplus(prLogVarAccum + Eps), Eps) * tf.abs(prValAccum - Labeli)) + tf.reduce_mean(tf.math.softplus(prLogVarAccum))
        lossUncAccum += loss_scale * lossUnc
    lossPhoto = lossOpticalAccum + lossUncAccum
    return lossPhoto

def DiceLoss(y_true, y_pred):
    y_pred = y_pred / 255.0
    y_true = y_true / 255.0
    smooth = 1e-05
    intersection = tf.compat.v1.reduce_sum(y_true * y_pred)
    union = tf.compat.v1.reduce_sum(y_true ** 2) + tf.compat.v1.reduce_sum(y_pred ** 2)
    dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice_coefficient
    return loss

def BCELoss(y_true, y_pred):
    y_pred = y_pred - 128
    y_true = y_true / 255.0
    bce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    return bce_loss

def LossDetail(prMB, LabelMB):
    loss = 128 * DiceLoss(prMB, LabelMB) + BCELoss(prMB, LabelMB)
    return loss

def LossMB(Label1PH, LabelMB01PH, prVal, Args):
    prValAccum = None
    prLogVarAccum = None
    prMBAccum = None
    loss_scales = [0.125, 0.25, 0.5, 1]
    Eps = 0.001
    for prVali, loss_scale in zip(prVal, loss_scales):
        if prValAccum == None:
            prValAccum = prVali[..., :Args.NumOut]
            Labeli = tf.compat.v1.image.resize(Label1PH, [prValAccum.shape[1], prValAccum.shape[2]])
            lossOpticali = tf.reduce_mean(tf.abs(prValAccum - Labeli))
            lossOpticalAccum = loss_scale * lossOpticali
            prLogVarAccum = prVali[..., Args.NumOut:2 * Args.NumOut]
            lossUnc = tf.reduce_mean(1 / tf.maximum(tf.math.softplus(prLogVarAccum + Eps), Eps) * tf.abs(prValAccum - Labeli)) + tf.reduce_mean(tf.math.softplus(prLogVarAccum))
            lossUncAccum = loss_scale * lossUnc
            prMBAccum = prVali[..., 2 * Args.NumOut:]
            LabelMB01i = tf.compat.v1.image.resize(LabelMB01PH, [prMBAccum.shape[1], prMBAccum.shape[2]])
            lossDetailAccum = loss_scale * LossDetail(prMBAccum, LabelMB01i)
            continue
        prValAccum = tf.compat.v1.image.resize(prValAccum, [prVali.shape[1], prVali.shape[2]])
        prValAccum += prVali[..., :Args.NumOut]
        Labeli = tf.compat.v1.image.resize(Label1PH, [prValAccum.shape[1], prValAccum.shape[2]])
        lossOpticali = tf.reduce_mean(tf.abs(prValAccum - Labeli))
        lossOpticalAccum += loss_scale * lossOpticali
        prLogVarAccum = tf.compat.v1.image.resize(prLogVarAccum, [prVali.shape[1], prVali.shape[2]])
        prLogVarAccum += prVali[..., Args.NumOut:2 * Args.NumOut]
        lossUnc = tf.reduce_mean(1 / tf.maximum(tf.math.softplus(prLogVarAccum + Eps), Eps) * tf.abs(prValAccum - Labeli)) + tf.reduce_mean(tf.math.softplus(prLogVarAccum))
        lossUncAccum += loss_scale * lossUnc
        prMBAccum = tf.compat.v1.image.resize(prMBAccum, [prVali.shape[1], prVali.shape[2]])
        prMBAccum += prVali[..., 2 * Args.NumOut:]
        LabelMB01i = tf.compat.v1.image.resize(LabelMB01PH, [prMBAccum.shape[1], prMBAccum.shape[2]])
        lossDetailAccum += loss_scale * LossDetail(prMBAccum, LabelMB01i)
    lossPhoto = lossOpticalAccum + lossUncAccum + lossDetailAccum
    return lossPhoto

@Scope
def Loss(I1PH, I2PH, Label1PH, Label2PH, prVal, Args):
    Args.UncType = 'LinearSoftplus'
    if Args.NetworkName == 'Network.MultiScaleMBResNet':
        return LossMB(Label1PH, prVal, Args)
    if Args.UncType == 'LinearSoftplus' and Args.LossFuncName == 'MultiscaleSL1-1':
        return LossMultiscaleUncertainity(Label1PH, prVal, Args)
    if Args.UncType == 'Aleatoric':
        eps = 1e-06
        MaxVal = 10.0
        prDisparity = prVal[:, :, :, :Args.NumOut]
        prLogvar = prVal[:, :, :, Args.NumOut:]
    elif Args.UncType == 'Inlier' or Args.UncType == 'LinearSoftplus':
        prDisparity = prVal[:, :, :, :Args.NumOut]
        prLogvar = prVal[:, :, :, Args.NumOut:]
    else:
        prDisparity = prVal
        prLogvar = None
        if Args.LossFuncName == 'SL2-1':
            lossPhoto = tf.reduce_mean(tf.square(prDisparity - Label1PH))
        if Args.LossFuncName == 'SL1-1':
            lossPhoto = tf.reduce_mean(tf.abs(prDisparity - Label1PH))
        if Args.LossFuncName == 'MultiscaleSL1-1':
            prValAccum = None
            lossAccum = None
            loss_scales = [0.125, 0.25, 0.5, 1]
            for prVali, loss_scale in zip(prVal, loss_scales):
                if prValAccum == None:
                    prValAccum = prVali
                    Labeli = tf.compat.v1.image.resize(Label1PH, [prValAccum.shape[1], prValAccum.shape[2]])
                    lossi = tf.reduce_mean(tf.abs(prValAccum - Labeli))
                    lossAccum = loss_scale * lossi
                    continue
                prValAccum = tf.compat.v1.image.resize(prValAccum, [prVali.shape[1], prVali.shape[2]])
                prValAccum += prVali
                Labeli = tf.compat.v1.image.resize(Label1PH, [prValAccum.shape[1], prValAccum.shape[2]])
                lossi = tf.reduce_mean(tf.abs(prValAccum - Labeli))
                lossAccum += loss_scale * lossi
            lossPhoto = lossAccum
        elif Args.LossFuncName == 'PhotoL1-1':
            DiffImg = prDisparity - Label1PH
            lossPhoto = tf.reduce_mean(tf.abs(DiffImg))
        elif Args.LossFuncName == 'PhotoChab-1':
            DiffImg = prDisparity - Label1PH
            epsilon = 0.001
            alpha = 0.45
            lossPhoto = tf.reduce_mean(tf.pow(tf.square(DiffImg) + tf.square(epsilon), alpha))
        elif Args.LossFuncName == 'SSIM-1':
            DiffImg = prDisparity - Label1PH
            AlphaSSIM = 0.005
            lossPhoto = tf.reduce_mean(tf.clip_by_value((1 - SSIM(prDisparity, Label1PH)) / 2, 0, 1) + AlphaSSIM * tf.abs(DiffImg))
        elif Args.LossFuncName == 'PhotoRobust':
            Epsa = 0.001
            c = 0.01
            DiffImg = WarpI1Patch - I2PH
            a = C2PH / 255.0
            a = tf.multiply(2.0 - 2.0 * Epsa, tf.math.sigmoid(a)) + Epsa
            lossPhoto = tf.reduce_mean(nll(DiffImg, a, c=c))
    if Args.RegFuncName == 'None':
        lossReg = 0.0
    else:
        print('Unknown Reg Func Type')
        sys.exit()
    if prLogvar is not None:
        if Args.UncType == 'Aleatoric':
            lossUnc = 0.5 * (tf.reduce_mean(tf.multiply(tf.exp(-prLogvar), tf.abs(prDisparity - Label1PH))) + tf.reduce_mean(prLogvar))
        if Args.UncType == 'Inlier':
            Lambda = 0.2
            InlierMask = prLogvar[:, :, :, 1]
            RefMask = np.tile(np.array([0, 1]), (Args.MiniBatchSize, Args.PatchSize[0], Args.PatchSize[1], 1))
            lossUnc = tf.reduce_mean(tf.multiply(tf.expand_dims(tf.nn.softmax(InlierMask), -1), tf.abs(prDisparity - Label1PH))) + Lambda * tf.reduce_mean(Lambda * tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(RefMask, [-1, 2]), logits=tf.reshape(prLogvar, [-1, 2])))
        if Args.UncType == 'LinearSoftplus':
            Eps = 0.001
            lossUnc = tf.reduce_mean(1 / tf.maximum(tf.math.softplus(prLogvar + Eps), Eps) * tf.abs(prDisparity - Label1PH)) + tf.reduce_mean(tf.math.softplus(prLogvar))
        return lossUnc + lossReg
    else:
        return lossPhoto + lossReg