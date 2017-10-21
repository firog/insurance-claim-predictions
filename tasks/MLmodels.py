import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb

def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses
     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)

def gini_score(estimator, train, target):
    prediction = estimator.predict_proba(train)
    return gini(target, prediction[:, 1]) / gini(target, target)

def logisticRegression(train, target, folds):
    model = LogisticRegression()
    predictions = cross_val_predict(model, train, target, cv=folds, method='predict_proba', n_jobs=2)
    model.fit(train, target)
    score = model.score(train, target)
    return (predictions, score, model)

def lgbboostGridSearch(train, target):
    model = lgb.LGBMClassifier()
    parameters = {
        #'nthread':[4], #when use hyperthread, xgboost may become slower
        'objective':['binary:logistic'],
        'learning_rate': [0.05], #so called `eta` value
        'max_depth': [5],
        #'min_child_weight': [11],
        #'silent': [1],
        'subsample': [0.7],
        'colsample_bytree': [0.3],
        'n_estimators': [500], #number of trees, change it to 1000 for better results
        #'missing':[-1],
        #'seed': [1337]
    }
    clf = GridSearchCV(model, parameters, n_jobs=8, cv=3,
                       scoring=gini_score,
                       verbose=2, refit=True)
    clf.fit(train, target)
    return clf

def xgboostGridSearch(train, target):
    xgb_model = XGBClassifier()
    parameters = {
        #'nthread':[4], #when use hyperthread, xgboost may become slower
        'objective':['binary:logistic'],
        'learning_rate': [0.05], #so called `eta` value
        'max_depth': [3, 10],
        #'min_child_weight': [11],
        #'silent': [1],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.3, 0.8],
        'n_estimators': [500], #number of trees, change it to 1000 for better results
        'missing':[-1],
        #'seed': [1337]
    }
    clf = GridSearchCV(xgb_model, parameters, n_jobs=8, cv=3,
                       scoring=gini_score,
                       verbose=2, refit=True)
    clf.fit(train, target)
    return clf

def xgboostPredict(train, target, test):
    model = XGBClassifier(max_depth=8)
    model.fit(train, target)
    predictions = model.predict_proba(test)
    return predictions

def lgboostPredict(train, target, test):
    model = lgb.LGBMClassifier(boosting_type='gbdt', colsample_bytree=1.0, learning_rate=0.05,
        max_bin=255, max_depth=10, min_child_samples=10,
        min_child_weight=5, min_split_gain=0.0, n_estimators=200,
        n_jobs=-1, num_leaves=31, objective='binary', random_state=0,
        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
        subsample_for_bin=50000, subsample_freq=1)
    model.fit(train, target)
    print('Feature importances:', list(model.feature_importances_))
    print(len(list(model.feature_importances_)))
    predictions = model.predict_proba(test)
    return predictions

def neuralNetPredicter(train, target, test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244,50,), random_state=1)
    clf.activation='logistic'
    clf.verbose=True
    clf.fit(train, target)
    predictions = clf.predict_proba(test)
    return predictions

def runPca(array):
    pca = PCA(n_components=array.shape[1])
    return pca.fit(array)
