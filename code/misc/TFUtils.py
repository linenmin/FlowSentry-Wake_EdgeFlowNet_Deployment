import os, pdb
import sys
import time
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
sys.dont_write_bytecode = True

def FindNumParams(PrintFlag=None):
    if PrintFlag is not None:
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])

def FindNumFlops(sess, PrintFlag=None):
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=tf.compat.v1.RunMetadata(), cmd='op', options=opts)
    if PrintFlag is not None:
        print('Number of Flops in this model are %d' % flops.total_float_ops)
    return flops.total_float_ops

def SetGPU(GPUNum=-1):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUNum)

def CalculateModelSize(PrintFlag=None):
    var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)]
    SizeMB = sum(var_sizes) / 1024.0 ** 2
    if PrintFlag is not None:
        print('Expected Model Size is %f' % SizeMB)
    return SizeMB

def Rename(CheckPointPath, ReplaceSource=None, ReplaceDestination=None, AddPrefix=None, AddSuffix=None):
    if not os.path.isdir(CheckPointPath):
        print('CheckPointsPath should be a directory!')
        os._exit(0)
    CheckPoint = tf.train.get_checkpoint_state(CheckPointPath)
    with tf.Session() as sess:
        for VarName, _ in tf.contrib.framework.list_variables(CheckPointPath):
            Var = tf.contrib.framework.load_variable(CheckPointPath, VarName)
            NewName = VarName
            if ReplaceSource is not None:
                NewName = NewName.replace(ReplaceSource, ReplaceDestination)
            if AddPrefix is not None:
                NewName = AddPrefix + NewName
            if AddSuffix is not None:
                NewName = NewName + AddSuffix
            print('Renaming %s to %s.' % (VarName, NewName))
            Var = tf.Variable(Var, name=NewName)
        Saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        Saver.save(sess, CheckPoint.model_checkpoint_path)

def PrintVars(CheckPointPath):
    if not os.path.isdir(CheckPointPath):
        print('CheckPointsPath should be a directory!')
        os._exit(0)
    CheckPoint = tf.train.get_checkpoint_state(CheckPointPath)
    with tf.Session() as sess:
        for VarName, _ in tf.contrib.framework.list_variables(CheckPointPath):
            print('%s' % VarName)

def freeze_graph(model_dir, output_node_names):
    if not tf.gfile.Exists(model_dir):
        raise AssertionError("Export directory doesn't exists. Please specify an export directory: %s" % model_dir)
    if not output_node_names:
        print('You need to supply the name of a node to --output_node_names.')
        return -1
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    absolute_model_dir = '/'.join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + '/frozen_model.pb'
    clear_devices = True
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), output_node_names.split(','))
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph.' % len(output_graph_def.node))
    return output_graph_def