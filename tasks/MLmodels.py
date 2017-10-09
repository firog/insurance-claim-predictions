from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def gini_score(estimator, train, target):
    prediction = estimator.predict_proba(train)
    return gini(target, prediction[:, 1]) / gini(target, target)


def logisticRegression(train, target, folds):
    model = LogisticRegression()
    predictions = cross_val_predict(model, train, target, cv=folds, method='predict_proba', n_jobs=2)
    model.fit(train, target)
    score = model.score(train, target)
    return (predictions, score, model)

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
    model = XGBClassifier(max_depth=8, )
    model.fit(train, target)
    predictions = model.predict_proba(test)
    return predictions

def neuralNetPredicter(train, target):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train, target)
    return clf

def runPca(array):
    pca = PCA(n_components=array.shape[1])
    return pca.fit(array)
