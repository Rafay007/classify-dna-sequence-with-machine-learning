import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
## Plot
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import matplotlib as plt
import xgboost as xgb
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
path='DNA_Dataset_Normalized.csv'
data=pd.read_csv(path)
print(data.head())
from sklearn.model_selection import KFold
y=data['Class']
x=data.drop(['Class'],axis=1)
x=StandardScaler().fit_transform(x)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
#

def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

xgb_preds = []

K = 5
kf = KFold(n_splits = K, random_state = 3228, shuffle = True)
for train_index, test_index in kf.split(x):
    train_X, valid_X = x[train_index], x[test_index]
    train_y, valid_y = x[train_index], x[test_index]

    # params configuration also from the1owl's kernel
    # https://www.kaggle.com/the1owl/forza-baseline
    xgb_params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
                  'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}

    d_train = xgb.DMatrix(train_X, train_y)
    d_valid = xgb.DMatrix(valid_X, valid_y)
    d_test = xgb.DMatrix(x)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 5000, watchlist, feval=gini_xgb, maximize=True, verbose_eval=50,
                      early_stopping_rounds=100)

    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))



preds = model.predict(x)
pred_probab=model.predict_proba(x)


preds=[]
for i in range(len(xgb_preds[0])):
    sum=0
    for j in range(K):
        sum+=xgb_preds[j][i]
    preds.append(sum / K)

output = pd.DataFrame({'target': preds})
output.to_csv("{}-foldCV_avg_sub.csv".format(K), index=False)
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score,roc_curve,auc,accuracy_score,recall_score,precision_score


















