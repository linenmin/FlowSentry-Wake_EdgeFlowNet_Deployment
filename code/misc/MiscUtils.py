import time
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import importlib
import re
sys.dont_write_bytecode = True

def tic():
    StartTime = time.time()
    return StartTime

def toc(StartTime):
    return time.time() - StartTime

def remap(x, oMin, oMax, iMin=None, iMax=None):
    if oMin == oMax:
        print('Warning: Zero output range')
        return None
    if iMin is None:
        iMin = np.amin(x)
    if iMax is None:
        iMax = np.amax(x)
    if iMin == iMax:
        print('Warning: Zero input range')
        return None
    result = np.add(np.divide(np.multiply(x - iMin, oMax - oMin), iMax - iMin), oMin)
    return result

def convertToOneHot(vector, n_labels):
    return np.equal.outer(vector, np.arange(n_labels)).astype(np.float)

def RotMatError(R1, R2):
    return np.linalg.norm(np.matmul(np.matrix.transpose(R1), R2) - np.eye(3), ord='fro')

def isRotMat(R):
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(np.matmul(R, R.transpose()), np.identity(R.shape[0], np.float), rtol=0.001, atol=0.001)
    should_be_one = np.allclose(np.linalg.det(R), 1.0, rtol=0.001, atol=0.001)
    return should_be_identity and should_be_one

def TransError(T1, T2):
    return np.linalg.norm(T1 - T2)

def ClosestRotMat(RDash):
    U, S, Vt = np.linalg.svd(RDash, full_matrices=False)
    SModified = np.eye(3)
    SModified[2, 2] = np.linalg.det(np.matmul(U, Vt))
    Rot = np.matmul(np.matmul(U, Vt), Vt)
    return Rot

def printcolor(s, color='r'):
    if color == 'r':
        print('\x1b[91m {}\x1b[00m'.format(s))
    elif color == 'g':
        print('\x1b[92m {}\x1b[00m'.format(s))
    elif color == 'y':
        print('\x1b[93m {}\x1b[00m'.format(s))
    elif color == 'lp':
        print('\x1b[94m {}\x1b[00m'.format(s))
    elif color == 'p':
        print('\x1b[95m {}\x1b[00m'.format(s))
    elif color == 'c':
        print('\x1b[96m {}\x1b[00m'.format(s))
    elif color == 'lgray':
        print('\x1b[97m {}\x1b[00m'.format(s))
    elif color == 'k':
        print('\x1b[98m {}\x1b[00m'.format(s))

def readPFM(file):
    file = open(file, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None
    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match('^(\\d+)\\s(\\d+)\\s$', file.readline().decode('ascii'))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().decode('ascii').rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return (data, scale)

def writePFM(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
    image = np.flipud(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))
    endian = image.dtype.byteorder
    if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
        scale = -scale
    file.write('%f\n'.encode() % scale)
    image.tofile(file)

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:, :, 0:2]
    f = open(name, 'rb')
    header = f.read(4)
    if header.decode('utf-8') != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape) == 3:
            return data[:, :, 0:3]
        else:
            return data
    return misc.imread(name)

def writeImage(name, data):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return writePFM(name, data, 1)
    return misc.imsave(name, data)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)

def readFloat(name):
    f = open(name, 'rb')
    if f.readline().decode('utf-8') != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)
    dim = int(f.readline())
    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d
    dims = list(reversed(dims))
    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))
    return data

def writeFloat(name, data):
    f = open(name, 'wb')
    dim = len(data.shape)
    if dim > 3:
        raise Exception('bad float file dimension: %d' % dim)
    f.write('float\n'.encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))
    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))
    data = data.astype(np.float32)
    if dim == 2:
        data.tofile(f)
    else:
        np.transpose(data, (2, 0, 1)).tofile(f)