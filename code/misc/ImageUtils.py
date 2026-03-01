import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import tensorflow as tf
import re
sys.dont_write_bytecode = True

def CenterCrop(I, OutShape):
    AppendFlag = False
    if len(np.shape(I)) == 3:
        I = I[np.newaxis, :, :, :]
        AppendFlag = True
    ImageSize = np.shape(I)
    CenterX = ImageSize[1] / 2
    CenterY = ImageSize[2] / 2
    try:
        ICrop = I[:, int(np.ceil(CenterX - OutShape[0] / 2)):int(np.ceil(CenterX + OutShape[0] / 2)), int(np.ceil(CenterY - OutShape[1] / 2)):int(np.ceil(CenterY + OutShape[1] / 2)), :]
        if AppendFlag:
            ICrop = np.squeeze(ICrop, axis=0)
    except:
        ICrop = None
    if OutShape[0] > ImageSize[1] or OutShape[1] > ImageSize[2]:
        ICrop = None
    return ICrop

def Give4Crops(I, axis=1):
    AppendFlag = False
    if len(np.shape(I)) == 3:
        I = I[np.newaxis, :, :, :]
        AppendFlag = True
    ImageSize = np.shape(I)
    CenterX = ImageSize[1] // 2
    CenterY = ImageSize[2] // 2
    ICrop1 = I[:, 0:CenterX, 0:CenterY, :]
    ICrop2 = I[:, CenterX:ImageSize[1], 0:CenterY, :]
    ICrop3 = I[:, 0:CenterX, CenterY:ImageSize[2], :]
    ICrop4 = I[:, CenterX:ImageSize[1], CenterY:ImageSize[2], :]
    if axis == 1:
        ICrops = np.stack([ICrop1, ICrop2, ICrop3, ICrop4], axis=1)
    elif axis == 2:
        ICrops = np.concatenate([ICrop1, ICrop2, ICrop3, ICrop4], axis=3)
    if AppendFlag:
        ICrops = np.squeeze(ICrops, axis=0)
    return ICrops

def OverlapCrop(I, Args, axis=1):
    AppendFlag = False
    if len(np.shape(I)) == 3:
        I = I[np.newaxis, :, :, :]
        AppendFlag = True
    ImageSize = np.shape(I)
    CenterX = ImageSize[1] // 2
    CenterY = ImageSize[2] // 2
    ICrop1 = I[:, 0:CenterX + Args.PatchDelta, 0:CenterY + Args.PatchDelta, :]
    ICrop2 = I[:, CenterX - Args.PatchDelta:ImageSize[1], 0:CenterY + Args.PatchDelta, :]
    ICrop3 = I[:, 0:CenterX + Args.PatchDelta, CenterY - Args.PatchDelta:ImageSize[2], :]
    ICrop4 = I[:, CenterX - Args.PatchDelta:ImageSize[1], CenterY - Args.PatchDelta:ImageSize[2], :]
    if axis == 1:
        ICrops = np.stack([ICrop1, ICrop2, ICrop3, ICrop4], axis=1)
    elif axis == 2:
        ICrops = np.concatenate([ICrop1, ICrop2, ICrop3, ICrop4], axis=3)
    if AppendFlag:
        ICrops = np.squeeze(ICrops, axis=0)
    return ICrops

def Give8Crops(I, axis=1):
    AppendFlag = False
    if len(np.shape(I)) == 3:
        I = I[np.newaxis, :, :, :]
        AppendFlag = True
    ImageSize = np.shape(I)
    CenterX = ImageSize[1] // 2
    CenterY = ImageSize[2] // 2
    ICrop1 = I[:, 0:CenterX, 0:CenterY, :]
    ICrop1 = Give4Crops(ICrop1)
    ICrop2 = I[:, CenterX:ImageSize[1], 0:CenterY, :]
    ICrop2 = Give4Crops(ICrop2)
    ICrop3 = I[:, 0:CenterX, CenterY:ImageSize[2], :]
    ICrop3 = Give4Crops(ICrop3)
    ICrop4 = I[:, CenterX:ImageSize[1], CenterY:ImageSize[2], :]
    ICrop4 = Give4Crops(ICrop4)
    if axis == 1:
        ICrops = np.stack([ICrop1, ICrop2, ICrop3, ICrop4], axis=1)
    elif axis == 2:
        ICrops = np.concatenate([ICrop1, ICrop2, ICrop3, ICrop4], axis=3)
    if AppendFlag:
        ICrops = np.squeeze(ICrops, axis=0)
    return ICrops

def PadOutside(I, OutShape):
    AppendFlag = False
    if len(np.shape(I)) == 3:
        I = I[np.newaxis, :, :, :]
        AppendFlag = True
    Output = np.zeros((np.shape(I)[0], OutShape[0], OutShape[1], OutShape[2]))
    ImageSize = np.shape(I)
    CenterX = OutShape[0] / 2
    CenterY = OutShape[1] / 2
    try:
        Output[:, int(np.ceil(CenterX - ImageSize[1] / 2)):int(np.ceil(CenterX + ImageSize[1] / 2)), int(np.ceil(CenterY - ImageSize[2] / 2)):int(np.ceil(CenterY + ImageSize[2] / 2)), :] = I
        if AppendFlag:
            Output = np.squeeze(Output, axis=0)
    except:
        Output = None
    return Output

def CenterCropFactor(I, Factor):
    AppendFlag = False
    if len(np.shape(I)) == 3:
        I = I[np.newaxis, :, :, :]
        AppendFlag = True
    ImageSize = np.array(np.shape(I))
    CenterX = ImageSize[1] / 2
    CenterY = ImageSize[2] / 2
    OutShape = ImageSize - np.mod(ImageSize, 2 ** Factor)
    OutShape[3] = ImageSize[3]
    try:
        ICrop = I[:, int(np.ceil(CenterX - OutShape[1] / 2)):int(np.ceil(CenterX + OutShape[1] / 2)), int(np.ceil(CenterY - OutShape[2] / 2)):int(np.ceil(CenterY + OutShape[2] / 2)), :]
        if AppendFlag:
            ICrop = np.squeeze(ICrop, axis=0)
    except:
        ICrop = None
        OutShape = None
    if OutShape[1] > ImageSize[1] or OutShape[2] > ImageSize[2]:
        ICrop = None
        OutShape = None
    return (ICrop, OutShape)

def RandomCrop(I, OutShape):
    AppendFlag = False
    if len(np.shape(I)) == 3:
        I = I[np.newaxis, :, :, :]
        AppendFlag = True
    ImageSize = np.shape(I)
    try:
        RandX = random.randint(0, ImageSize[1] - OutShape[0])
        RandY = random.randint(0, ImageSize[2] - OutShape[1])
        ICrop = I[:, RandX:RandX + OutShape[0], RandY:RandY + OutShape[1], :]
        if AppendFlag:
            ICrop = np.squeeze(ICrop, axis=0)
    except:
        ICrop = None
    return ICrop

def StackImages(I1, I2):
    return np.dstack((I1, I2))

def UnstackImages(I, NumChannels=3):
    return (I[:, :, :NumChannels], I[:, :, NumChannels:])

class DataAugmentationTF:

    def __init__(self, sess, ImgPH, Augmentations=['Brightness', 'Contrast', 'Hue', 'Saturation', 'Gamma', 'Gaussian']):
        self.Augmentations = Augmentations
        self.ImgPH = ImgPH
        self.sess = sess

    def RandPerturbBatch(self):
        IRet = self.ImgPH
        for perturb in self.Augmentations:
            if perturb == 'Brightness':
                IRet = tf.clip_by_value(tf.image.random_brightness(IRet, max_delta=20), 0.0, 255.0)
            elif perturb == 'Contrast':
                IRet = tf.clip_by_value(tf.image.random_contrast(IRet, lower=0.5, upper=1.5), 0.0, 255.0)
            elif perturb == 'Hue':
                IRet = tf.clip_by_value(tf.image.random_hue(IRet, max_delta=0.5), 0.0, 255.0)
            elif perturb == 'Saturation':
                IRet = tf.clip_by_value(tf.image.random_saturation(IRet, lower=0.5, upper=1.5), 0.0, 255.0)
            elif perturb == 'Gamma':
                IRet = tf.clip_by_value(tf.image.adjust_gamma(IRet, gamma=np.random.uniform(low=0.9, high=1.1), gain=1), 0.0, 255.0)
            elif perturb == 'Gaussian':
                IRet = tf.clip_by_value(IRet + tf.random.normal(shape=tf.shape(IRet), mean=0.0, stddev=20.0), 0.0, 255.0)
        return IRet

def Resize(I, OutShape):
    ImageSize = np.shape(I)
    IResize = cv2.resize(I, (OutShape[1], OutShape[0]))
    return IResize

def StandardizeInputs(I):
    I /= 255.0
    I -= 0.5
    I *= 2.0
    return I

def StandardizeInputsTF(I):
    I = tf.math.multiply(tf.math.subtract(tf.math.divide(I, 255.0), 0.5), 2.0)
    return I

def HPFilter(I, Radius=10):
    if len(np.shape(I)) == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    F = cv2.dft(np.float32(I), flags=cv2.DFT_COMPLEX_OUTPUT)
    FShift = np.fft.fftshift(F)
    Rows, Cols = I.shape
    Mask = np.ones((Rows, Cols, 2), np.uint8)
    Center = [int(Rows / 2), int(Cols / 2)]
    x, y = np.ogrid[:Rows, :Cols]
    MaskArea = (x - Center[0]) ** 2 + (y - Center[1]) ** 2 <= Radius ** 2
    Mask[MaskArea] = 0
    FShiftFilt = np.multiply(FShift, Mask)
    FFilt = np.fft.ifftshift(FShiftFilt)
    IFilt = cv2.idft(FFilt)
    IFilt = cv2.magnitude(IFilt[:, :, 0], IFilt[:, :, 1])
    IFilt = np.tile(IFilt[:, :, np.newaxis], (1, 1, 3))
    return IFilt

def rgb2gray(rgb):
    AppendFlag = False
    if len(np.shape(rgb)) == 3:
        rgb = rgb[np.newaxis, :, :, :]
        AppendFlag = True
    r, g, b = (rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2])
    gray = 0.2989 * r + 0.587 * g + 0.114 * b
    if AppendFlag:
        gray = np.squeeze(gray, axis=0)
    return gray

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
        return (None, None)
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

def calculate_aspect_ratio(resolution):
    return resolution[0] / resolution[1]

def closest_resizing(current_res, target_res):
    current_aspect_ratio = calculate_aspect_ratio(current_res)
    target_aspect_ratio = calculate_aspect_ratio(target_res)
    if target_aspect_ratio < current_aspect_ratio:
        new_height = target_res[1]
        new_width = int(new_height * current_aspect_ratio)
    else:
        new_width = target_res[0]
        new_height = int(new_width / current_aspect_ratio)
    return ((current_aspect_ratio, target_aspect_ratio), (new_width, new_height))

def ResizeNearestCrop(image, target_shape):
    nearest_resolution = closest_resizing((image.shape[1], image.shape[0]), (target_shape[1], target_shape[0]))[1]
    resized_image = Resize(image, (nearest_resolution[1], nearest_resolution[0]))
    cropped_image = CenterCrop(resized_image, target_shape)
    return cropped_image

def GiveNCrops(I, halves=1, axis=1):
    AppendFlag = False
    if len(np.shape(I)) == 3:
        I = I[np.newaxis, :, :, :]
        AppendFlag = True
    ImageSize = np.shape(I)
    CenterX = ImageSize[1] // 2
    CenterY = ImageSize[2] // 2
    if halves == 0:
        return I[:, :, :, :]
    else:
        halves -= 1
        ICrop1 = GiveNCrops(I[:, 0:CenterX, 0:CenterY, :], halves, axis)
        ICrop2 = GiveNCrops(I[:, 0:CenterX, CenterY:ImageSize[2], :], halves, axis)
        ICrop3 = GiveNCrops(I[:, CenterX:ImageSize[1], 0:CenterY, :], halves, axis)
        ICrop4 = GiveNCrops(I[:, CenterX:ImageSize[1], CenterY:ImageSize[2], :], halves, axis)
        if axis == 1 and halves == 0:
            ICrops = np.stack([ICrop1, ICrop2, ICrop3, ICrop4], axis=1)
        elif axis == 1 and halves != 0:
            ICrops = np.concatenate([ICrop1, ICrop2, ICrop3, ICrop4], axis=1)
        elif axis == 2:
            ICrops = np.concatenate([ICrop1, ICrop2, ICrop3, ICrop4], axis=3)
    if AppendFlag:
        ICrops = np.squeeze(ICrops, axis=0)
    return ICrops

def split_image_into_chunks(image, splits):
    if splits <= 0:
        raise ValueError('Number of splits must be greater than 0.')
    M, N = image.shape[:2]
    chunk_size_m = int(M / np.sqrt(splits))
    chunk_size_n = int(N / np.sqrt(splits))
    chunks = []
    for i in range(int(np.sqrt(splits))):
        for j in range(int(np.sqrt(splits))):
            chunk = image[i * chunk_size_m:(i + 1) * chunk_size_m, j * chunk_size_n:(j + 1) * chunk_size_n, ...]
            chunks.append(chunk)
    chunks = np.stack(chunks, axis=0)
    return chunks