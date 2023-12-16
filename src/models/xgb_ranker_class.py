
import pandas as pd
from xgboost import XGBRanker, XGBClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import class_weight



class MyXGBRanker(XGBRanker, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        cdf = x.groupby('era').agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x[x.columns[1:]], y, group=group)

    def predict(self, x):
        return super().predict(x[x.columns[:]])


class XGBRanker_PIPE(XGBRanker, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        x=pd.DataFrame(x)
        #x.columns = range(x.shape[1])
        cdf = x.groupby(0).agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x.iloc[:,1:], y, group=group)

    def predict(self, x):
        x=pd.DataFrame(x)
        #x.columns = range(x.shape[1])
        return super().predict(x.iloc[:,1:])


class XGBRanker_ORIG(XGBRanker):
    def fit(self, x, y):
        cdf = x.groupby('era').agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x[features], y, group=group)

    def predict(self, x):
        return super().predict(x[features])        


class MyXGBClass(XGBClassifier, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        return super().fit(x, y,sample_weight=class_weight.compute_sample_weight(class_weight='balanced', y=y))

    def predict(self, x):
        return super().predict(x)


class MyXGBRanker_l1(XGBRanker, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        cdf = x.groupby('era').agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x[x.columns[1:]], y, group=group)

    def predict(self, x):
        return super().predict(x[x.columns[1:]])
