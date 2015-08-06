from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from scipy.stats import rankdata
class EnsembleRegressor(BaseEstimator, RegressorMixin):
	def __init__(self, regressors, pooling='average'):
		self.regressors = regressors
		self.pooling = pooling

	def fit(self, X, y):
		for reg in self.regressors:
			reg.ift(X, y)

	def predict(self, X):
		pred_ = np.zeros(X.shape[0])
		for reg in self.regressors:
			if self.pooling == 'average':
				pred_ += reg.predict(X)
			elif self.pooling == 'rank_average':
				pred_ + = rankdata(reg.predict(X))
			else:
				raise ValueError('Unknown pooling method')

		return pred_ / len(self.regressors)
