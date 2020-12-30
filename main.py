import numpy as np

# import NaviBayes as nb

trainTestRate = 0.7  # 测试比例
filePath = "./iris.csv"  # 数据集路径


# 导入数据
def loadData(Path):
    fileLines = open(Path, "r").readlines()

    trainLines = 50 * trainTestRate

    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []

    linesCount = 1
    # 根据比例设置训练集和测试集的数量，数据时随机分布的，所以连续取出即可
    for line in fileLines:
        currLine = line.strip().split(',')
        lineArr = []
        for i in range(4):
            lineArr.append(float(currLine[i]))
        if linesCount <= trainLines:
            trainingSet.append(lineArr)
            trainingLabels.append(int(currLine[-1]))
            linesCount += 1
        else:
            testSet.append(lineArr)
            testLabels.append(int(currLine[-1]))
            linesCount += 1

        if linesCount > 50:
            linesCount = 1

    return trainingSet, trainingLabels, testSet, testLabels


# 数据归1化
def normalization(inX):
    minVals = inX.min(0)
    maxVals = inX.max(0)
    ranges = maxVals - minVals
    normDataset = np.zeros(np.shape(inX))
    m = inX.shape[0]
    normDataset = inX - np.tile(minVals, (m, 1))
    normDataset = normDataset / np.tile(ranges, (m, 1))
    return normDataset


if __name__ == '__main__':
    TrainingSet, TrainingLabels, TestSet, TestLabels = loadData(filePath)
    # 转成numpy数组
    TrainingSet1 = np.array(TrainingSet, dtype=float)
    TrainingLabels1 = np.array(TrainingLabels, dtype=int)
    TestSet1 = np.array(TestSet, dtype=float)
    TestLabels1 = np.array(TestLabels, dtype=int)
    # 特征归一化
    TrainingSet1 = normalization(TrainingSet1)
    TestSet1 = normalization(TestSet1)

    # 保存过程文件
    np.savetxt("TrainingSet1.txt", TrainingSet1, fmt="%.4f", delimiter=',')
    np.savetxt("TrainingLabels1.txt", TrainingLabels1, fmt="%d", delimiter=',')
    np.savetxt("TestSet1.txt", TestSet1, fmt="%.4f", delimiter=',')
    np.savetxt("TestLabels1.txt", TestLabels1, fmt="%d", delimiter=',')


