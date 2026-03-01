import random

def RandSimilarityPerturbationNP(I1, HObj, PatchSize, ImageSize=None, Vis=False):
    if ImageSize is None:
        ImageSize = np.array(np.shape(I1))
    H, Params = HObj.GetRandReducedHICSTN(TransformType='psuedosimilarity')
    opt = warp2.Options(PatchSize=ImageSize, MiniBatchSize=1, warpType=['pseudosimilarity'])
    Params = np.squeeze(Params)
    H = np.squeeze(H)
    I2 = warp2.transformImageNP(opt, I1[np.newaxis, :, :, :], H[np.newaxis, :, :])[0]
    P1 = iu.CenterCrop(I1, PatchSize)
    P2 = iu.CenterCrop(I2, PatchSize)
    if Vis is True:
        cv2.imshow('I1, I2', np.hstack((I1, I2)))
        cv2.imshow('I1, I2', np.hstack((P1, P2)))
        cv2.waitKey(0)
    return (I1, I2, P1, P2, H, Params)

def GenerateBatchNP(TrainNames, PatchSize, MiniBatchSize, HObj, BasePath, OriginalImageSize):
    IBatch = []
    I1Batch = []
    I2Batch = []
    HBatch = []
    ParamsBatch = []
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        RandIdx = random.randint(0, len(TrainNames) - 1)
        RandImageName = BasePath + os.sep + TrainNames[RandIdx]
        I = cv2.imread(RandImageName)
        I = iu.RandomCrop(I, OriginalImageSize)
        if I is None:
            continue
        ImageNum += 1
        I1, I2, P1, P2, H, Params = RandSimilarityPerturbation(I, HObj, PatchSize, ImageSize=None, Vis=False)
        ICombined = np.dstack((P1[:, :, 0:3], P2[:, :, 0:3]))
        IS = iu.StandardizeInputs(np.float32(ICombined))
        IBatch.append(IS)
        I1Batch.append(P1)
        I2Batch.append(P2)
        HBatch.append(H)
        ParamsBatch.append(Params)
    return (IBatch, I1Batch, I2Batch, HBatch, ParamsBatch)