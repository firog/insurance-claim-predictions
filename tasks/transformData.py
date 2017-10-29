import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer

def loadData(path):
    df = pd.read_csv(path, na_values=-1)
    return df

def extractTarget(dataFrame):
    Ytarget = dataFrame["target"].values
    del dataFrame["target"]
    del dataFrame["id"]
    return Ytarget

def dataExtractor(dataFrame):
    Xtrain = dataFrame.values
    return Xtrain

def catBinExtractor(headers):
    catBinHeaders = []
    for headers in headers:
        if headers.endswith("_cat") or headers.endswith("_bin"):
            catBinHeaders.append(headers)
    return catBinHeaders

def oneHotEncode(df):
    enc = OneHotEncoder()
    enc.fit(df)
    onehotlabels = enc.transform(df).toarray()
    return onehotlabels

def imputer(array):
    imp = Imputer(missing_values=-1, strategy='mean', axis=0)
    newArray = imp.fit_transform(array)
    return newArray

def missingValCounter(df):
    count = (df == -1).astype(int).sum(axis=0)
    return count

def dataNormalizer(array):
    normalizer = Normalizer(norm='max')
    normArray = normalizer.fit_transform(array)
    return normArray
