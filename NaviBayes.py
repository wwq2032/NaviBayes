from numpy import *

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Demo = 2.0
    p1Demo = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Demo += sum(trainMatrix[i])

        else:
            p0Num += trainMatrix[i]
            p0Demo += sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Demo)
    p0Vect = log(p0Num / p0Demo)

    return p0Vect, p1Vect, pAbusive



if __name__ == '__main__':

