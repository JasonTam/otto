from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import StratifiedKFold

import numpy as np


class FitlessMixin(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


class IdentityTformer(FitlessMixin):
    def transform(self, X, y=None, **fit_params):
        return X


class DenseTformer(FitlessMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()


class LogTformer(FitlessMixin):
    def transform(self, X, y=None, **fit_params):
        return np.log(X+1)


class NzTformer(FitlessMixin):
    def transform(self, X, y=None, **fit_params):
        return np.apply_along_axis(func1d=np.count_nonzero, 
                                   axis=1, arr=X)[:, None]


class NzvarTformer(FitlessMixin):
    def transform(self, X, y=None, **fit_params):
        ret = np.apply_along_axis(func1d=lambda r: np.var(r[np.nonzero(r)]), 
                                   axis=1, arr=X)[:, None]
        ret[np.isnan(ret)] = 0
        return ret


class NzmeanTformer(FitlessMixin):
    def transform(self, X, y=None, **fit_params):
        ret = np.apply_along_axis(func1d=lambda r: np.mean(r[np.nonzero(r)]), 
                                   axis=1, arr=X)[:, None]
        ret[np.isnan(ret)] = 0
        return ret
 

    