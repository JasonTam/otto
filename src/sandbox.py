import os
import numpy as np
from time import time
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
import io_tools as iot

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding

try:
    ids
except NameError:
    ids = np.load(os.path.join(iot.DATA_DIR, 'train_ids.npy'))
    feats = np.load(os.path.join(iot.DATA_DIR, 'train_feats.npy')).astype(float)
    labels = np.load(os.path.join(iot.DATA_DIR, 'train_labels_enc.npy'))

class FitlessMixin(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

class DenseTransformer(FitlessMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()
    
class LogTransformer(FitlessMixin):
    def transform(self, X, y=None, **fit_params):
        return np.log(X+1)

class HadamardTransformer(FitlessMixin):
    def transform(self, X, y=None, **fit_params):
		
		
        return np.log(X+1)


tfidf = TfidfTransformer().fit(feats)

        

skf = StratifiedKFold(labels, n_folds=3)
pipe = make_pipeline(
                     #~ TfidfTransformer(norm=u'l2', 
                                      #~ use_idf=True, 
                                      #~ smooth_idf=True, 
                                      #~ sublinear_tf=True),
                     DenseTransformer(),
                     StandardScaler(),
                     #~ LocallyLinearEmbedding(
                                #~ n_components=2, 
                                #~ n_neighbors=5,
                                #~ method='modified',
                                #~ eigen_solver='dense'),
                     #~ Isomap(n_components=10, n_neighbors=5),
                     #~ RandomForestClassifier(n_estimators=100, n_jobs=-1))
                     LogisticRegression())

tic = time()
preds = cross_val_predict(pipe, feats, labels, cv=skf, n_jobs=-1, probability=True)
toc = time() - tic

print pipe
print 'Time: %s s' % toc
print 'Log Loss:', log_loss(labels, preds)










