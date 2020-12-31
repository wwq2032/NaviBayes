import numpy as np
import logging
# import colorlog
import NaviBayes as nb

trainTestRate = 0.7  # 测试比例
filePath = "./iris.csv"  # 数据集路径
logging.basicConfig(filename="RECORD.log", filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lofConfig():
    # 调用模块时,如果错误引用，比如多次调用，每次会添加Handler，造成重复日志，这边每次都移除掉所有的handler，后面在重新添加，可以解决这类问题
    while logger.hasHandlers():
        for i in logger.handlers:
            logger.removeHandler(i)

    # file log 写入文件配置
    formatter = logging.Formatter('%(asctime)s - %(pathname)s: [line:%(lineno)d] - %(levelname)s: %(message)s')  # 日志的格式
    fh = logging.FileHandler(filename='RECORD.log', encoding='utf-8')  # 日志文件路径文件名称，编码格式
    fh.setLevel(logging.INFO)  # 日志打印级别
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    # console log 控制台输出控制
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# 导入数据
def loadData(Path):
    fileLines = open(Path, "r").readlines()

    trainLines = 50 * trainTestRate

    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []

    linesCount = 1

    # 根据比例设置训练集和测试集的数量，因为数据是随机分布的，所以连续取出即可
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


# 根据概率大小分类
def classify(inProbabMat):
    # 按列比较概率大小，返回最大值的行号索引，即类别编号
    temp = inProbabMat.argmax(axis=1)
    rresult = temp.reshape(temp.shape[0], 1)
    return rresult


# 计算分类精度
def getAccuracy(predict, reality):
    count = 0
    mistakeArr = []
    for i in range(predict.shape[0]):
        if predict[i] == reality[i]:
            count += 1
            logger.info("编号%d预测真确！预测值为：%d，真实值为：%d", i, predict[i], reality[i])
        else:
            logger.info("编号%d预测错误！预测值为：%d，真实值为：%d", i, predict[i], reality[i])
            mistakeArr.append(i)

    accuracy = count / predict.shape[0]
    return accuracy, mistakeArr


if __name__ == '__main__':
    lofConfig()
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
    np.savetxt("TrainingSet1.txt", TrainingSet1, fmt="%f", delimiter=',')
    np.savetxt("TrainingLabels1.txt", TrainingLabels1, fmt="%d", delimiter=',')
    np.savetxt("TestSet1.txt", TestSet1, fmt="%f", delimiter=',')
    np.savetxt("TestLabels1.txt", TestLabels1, fmt="%d", delimiter=',')

    meansMat, stdMat = nb.getMeansAndVar(TrainingSet1)

    logger.info("开始计算不同类别的概率")

    rows=round(150*(1-trainTestRate))
    resultProbab = np.zeros((rows, 3))
    for i in range(3):
        meansVec = meansMat[i, :]
        stdVec = stdMat[i, :]
        probabTemp = nb.calProbability(TestSet1, meansVec, stdVec)
        resultProbab[:, i] = probabTemp[:, 0]

    result = classify(resultProbab)
    np.savetxt("result.txt", result, fmt="%d", delimiter=',')
    accuracy, mistakeArr = getAccuracy(result, TestLabels1)

    logger.info("测试结果：共%d个测试样本，正确：%d个，错误：%d个，精度：%.3f%%", TestLabels1.shape[0], TestLabels1.shape[0] - len(mistakeArr),
                len(mistakeArr), 100 * accuracy)
    logger.info("预测错误的编号为 %s", mistakeArr)
