import pandas as pd
import numpy as np
from tasks.submit import generate_submission
from tasks.gini import gini_normalized
from tasks.transformData import loadData, extractTarget, dataExtractor, catBinExtractor, oneHotEncode, dataNormalizer
from tasks.transformData import imputer
from tasks.MLmodels import logisticRegression, xgboostGridSearch, xgboostPredict, lgboostPredict, neuralNetPredicter, runPca, lgbboostGridSearch
from sklearn.metrics import roc_curve, confusion_matrix

if __name__ == '__main__':
    print("Loading data")
    dfTrain = loadData("data/train.csv")
    dfTest = loadData("data/test.csv")
    Ytarget = extractTarget(dfTrain)
    calcCols = dfTrain.columns[dfTrain.columns.str.startswith('ps_calc_')]
    dfTrain = dfTrain.drop(calcCols, axis=1)
    dfTest = dfTest.drop(calcCols, axis=1)

    del dfTest["id"]
    # trainHeaders = dfTrain.columns
    # trainCatBinHeader = catBinExtractor(trainHeaders)
    # dfCatBinTrain = dfTrain[trainCatBinHeader] + 1
    # dfTrain = dfTrain.drop(trainCatBinHeader, axis = 1)

    # testHeaders = dfTest.columns
    # testCatBinHeader = catBinExtractor(testHeaders)
    # dfCatBinTest = dfTest[testCatBinHeader] + 1
    # dfTest = dfTest.drop(testCatBinHeader, axis = 1)
    
    Xtrain = dataExtractor(dfTrain)
    Xtest = dataExtractor(dfTest)

    # oneHotTest = oneHotEncode(dfCatBinTest)
    # oneHotTrain = oneHotEncode(dfCatBinTrain)
    # print("Concatenating train")
    # Xtrain = np.concatenate((Xtrain, oneHotTrain), axis = 1)
    # print("Concatenating test")
    # Xtest = np.concatenate((Xtest, oneHotTest), axis = 1)
    # print("Imputing train")
    # Xtrain = imputer(Xtrain)
    # print("Imputing test")
    # Xtest = imputer(Xtest)
    # print("Normalizing train")
    # Xtrain = dataNormalizer(Xtrain)
    # print("Normalizing test")
    # Xtest = dataNormalizer(Xtest)
    # print("Training model using logistic regression")
    # (predictions, score, model) = logisticRegression(Xtrain, Ytarget, 3)
    # predictionsLR = logisticRegression(Xtrain, Ytarget, 3)
    # predictionsXG = xgboostPredict(Xtrain, Ytarget, Xtest)
    # predictionsGrid = xgboostGridSearch(Xtrain, Xtarget)
    # print("best estimator")
    # print(predictionsGrid.best_estimator_)
    # print("best score")
    # print(predictionsGrid.best_score_)
    # pca1 = runPca(Xtrain)
    # print(pca1)
    # print(pca1.explained_variance_)
    # print(len(pca1.explained_variance_))
    # print("best parameters")
    # print(predictionsGrid.best_params_)
    # print("neural net pred")
    predictionsLGB = lgboostPredict(Xtrain, Ytarget, Xtest)
    # lgbGridSearch = lgbboostGridSearch(Xtrain, Ytarget)
    # predictionsNN = neuralNetPredicter(Xtrain, Ytarget, Xtest)
    # print(gini_normalized(Ytarget, predictionsXG[:, 1]))
    ids = loadData("data/test.csv")
    ids = ids["id"]
    generate_submission(ids, predictionsLGB)
