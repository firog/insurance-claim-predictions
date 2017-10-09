import pandas as pd
import numpy as np
from tasks.gini import gini_normalized
from tasks.transformData import loadData, extractTarget, dataExtractor, catBinExtractor, oneHotEncode, dataNormalizer
from tasks.transformData import imputer
from tasks.MLmodels import logisticRegression, xgboostGridSearch, xgboostPredict
from sklearn.metrics import roc_curve

if __name__ == '__main__':
    print("Loading data")
    dfTrain = loadData("data/train.csv")
    dfTest = loadData("data/test.csv")
    Ytarget = extractTarget(dfTrain)
 
    del dfTest["id"]
 
    trainHeaders = dfTrain.columns
    trainCatBinHeader = catBinExtractor(trainHeaders)
    dfCatBinTrain = dfTrain[trainCatBinHeader] + 1
    dfTrain = dfTrain.drop(trainCatBinHeader, axis = 1)

    testHeaders = dfTest.columns
    testCatBinHeader = catBinExtractor(testHeaders)
    dfCatBinTest = dfTest[testCatBinHeader] + 1
    dfTest = dfTest.drop(testCatBinHeader, axis = 1)

    Xtrain = dataExtractor(dfTrain)
    Xtest = dataExtractor(dfTest)

    oneHotTest = oneHotEncode(dfCatBinTest)
    oneHotTrain = oneHotEncode(dfCatBinTrain)
    print("Concatenating train")
    Xtrain = np.concatenate((Xtrain, oneHotTrain), axis = 1)
    print("Concatenating test")
    Xtest = np.concatenate((Xtest, oneHotTest), axis = 1)
    print("Imputing train")
    Xtrain = imputer(Xtrain)
    print("Imputing test")
    Xtest = imputer(Xtest)
    print("Normalizing train")
    Xtrain = dataNormalizer(Xtrain)
    print("Normalizing test")
    Xtest = dataNormalizer(Xtest)
    # print("Training model using logistic regression")
    # (predictions, score, model) = logisticRegression(Xtrain, Ytarget, 3)
    # predictionsLR = logisticRegression(Xtrain, Ytarget, 3)
    # predictionsXG = xgboostPredict(Xtrain, Ytarget, Xtest)
    predictionsGrid = xgboostGridSearch(Xtrain, Xtest)
    print(predictionsGrid.best)

    # print(gini_normalized(Ytarget, predictions[:, 1]))
