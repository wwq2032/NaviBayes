import numpy as np


def getMeansAndVar(inX):
    meansMat = np.zeros((3, 4), dtype=float)
    stdMat = np.zeros((3, 4), dtype=float)
    t = inX[0:35, ]
    meansMat[0] = np.mean(t, axis=0)
    stdMat[0] = np.std(t, axis=0)
    t = inX[35:70, ]
    meansMat[1] = np.mean(t, axis=0)
    stdMat[1] = np.std(t, axis=0)
    t = inX[70:105, ]
    meansMat[2] = np.mean(t, axis=0)
    stdMat[2] = np.std(t, axis=0)
    return meansMat, stdMat


def calProbability(noLabelMat, meansVec, stdVec):
    meansMat = np.tile(meansVec, (noLabelMat.shape[0], 1))
    stdMat = np.tile(stdVec, (noLabelMat.shape[0], 1))

    deltaX2 = (noLabelMat - meansMat) ** 2
    stdMat2 = stdMat ** 2
    k = 1 / (np.sqrt(2 * np.pi) * stdMat)

    # 各个特征的条件概率矩阵,或log优化去掉e的幂函数求和
    probabilityEveryFeature = k * np.exp(-deltaX2 / stdMat2)  # 每个特征的概率矩阵 15*4

    ######################################################################################
    # m = np.ones((4, 1))
    # 总条件概率矩阵（特征条件概率矩阵累加）
    # result = np.dot(probabilityEveryFeature, m)  # 应该是连乘不是累加，不过都是比大小，影响不大
    # return result
    ######################################################################################

    # 严格按照高斯分布概率公式，可能会产生下溢出问题，连乘之后都是0
    result = np.prod(probabilityEveryFeature, axis=1)  # 向量按列连乘
    rresult = result.reshape(result.shape[0], 1)

    return rresult
