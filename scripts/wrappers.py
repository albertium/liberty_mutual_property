import os
import shutil
import subprocess

import re
import copy

import numpy as np
from numpy.linalg import eig
from scipy.stats import rankdata

import pandas as pd

import gini

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, RegressorMixin
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso

import xgboost as xgb

# ====================================================================
# helpers
def pd2svm(filename, data):
    if isinstance(data, pd.DataFrame):
        data = data.values
    else:
        assert isinstance(data, np.ndarray), \
                "Data must be DataFrame or ndarray"

    last = data.shape[1]
    frmt = [str(i) + ':{}' for i in range(last)]
    frmt = ' '.join(frmt) + '\n'

    with open(filename, 'w') as f:
        for row in data:
            f.write(frmt.format(*row))

    return data.shape[1]


class Whitener(BaseEstimator, ClusterMixin, TransformerMixin):
	def __init__(self, var_e=1E-5, eig_e=1E-5):
		self.var_e = var_e
		self.eig_e = eig_e


	def fit(self, X):
		if isinstance(X, pd.DataFrame):
			X = X.values
		else:
			assert isinstance(X, np.ndarray), "Data must be DataFrame or ndarray"

		self.means = X.mean(axis=0)
		# e = (X.max(axis=0) - X.min(axis=0)) * self.var_e 
		self.stds = np.sqrt(X.var(axis=0) + self.var_e)

		X = (X - self.means) / self.stds
		cov = np.cov(X.T)
		D, self.V = eig(cov)
		self.D_sqrt = np.sqrt(D + self.eig_e)
		

	def transform(self, X):
		if isinstance(X, pd.DataFrame):
			X = X.values
		else:
			assert isinstance(X, np.ndarray), "Data must be DataFrame or ndarray"

		X = (X - self.means) / self.stds
		return np.dot(X, np.dot(self.V / self.D_sqrt, self.V.T))


	def fit_transform(self, X):
		if isinstance(X, pd.DataFrame):
			X = X.values
		else:
			assert isinstance(X, np.ndarray), "Data must be DataFrame or ndarray"
		
		self.means = X.mean(axis=0)
		# e = (X.max(axis=0) - X.min(axis=0)) * self.var_e 
		self.stds = np.sqrt(X.var(axis=0) + self.var_e)

		X = (X - self.means) / self.stds
		cov = np.cov(X.T)
		D, self.V = eig(cov)
		self.D_sqrt = np.sqrt(D + self.eig_e)

		return np.dot(X, np.dot(self.V / self.D_sqrt, self.V.T))


def generatePrimes(n):
	sieve = np.ones(n/3 + (n%6==2), dtype=np.bool)
	sieve[0] = False
	for i in range(int(n**0.5/3)+1):
		if sieve[i]:
			k=3*i+1|1
			sieve[((k*k)/3)::2*k] = False
			sieve[(k*k+4*k-2*k*(i&1))/3::2*k] = False
													
	for x in np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]:
		yield x


def calPowerCount(X, degree=2):
	X = X.copy()
	prime = generatePrimes(200000)

	# encode levels with prime numbers
	for col in X.columns:
		mapping = {level: next(prime) for level in X[col].unique()}
		X[col] = X[col].map(mapping)

	# calculate Power terms
	a = X.values
	b = X.values[:, :, np.newaxis]
	for i in range(degree-1):
		a = (a[:, np.newaxis, :] * b).reshape(a.shape[0], -1)

	# remove duplicates
	a = a.T
	b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
	a = np.unique(b).view(a.dtype).reshape(-1, a.shape[1]).T

	# get count
	a = pd.DataFrame(a)
	for col in a.columns:
		mapping = a[col].value_counts().to_dict()
		a[col] = a[col].map(mapping)

	a = a.rename(columns=lambda x: 'count_' + str(degree) + 'way_' + str(x))
	return a


# ====================================================================
# sofia wrapper
class sofia_kmeans(BaseEstimator, ClusterMixin, TransformerMixin):
	"""
	wrapper for sofia kmean via harddrive
	"""
	def __init__(self, name='sofia_0', n_clusters=10, iterations=1000, batch_size=100, mapping_param=0.5, mapping_threshold=0.01):
		self.command = ['sofia-kmeans',
		 				'--k', str(n_clusters),
						'--init_type', ' optimized_kmeans_pp',
						'--opt_type', 'mini_batch_kmeans',
						'--mini_batch_size', str(batch_size),
						'--iterations', str(iterations),
						'--objective_after_training']
		self.command = ' '.join(self.command)

		self.mapping_param = mapping_param
		self.mapping_threshold = mapping_threshold
		self.path = '../tmp/' + name + '/'
		self.whitener = Whitener()

		if not os.path.exists(self.path):
			os.makedirs(self.path)
		else:
			shutil.rmtree(self.path)
			os.makedirs(self.path)


	def _fit(self, filename, dimensionality):
		self.command_train = self.command \
				+ ' --training_file ' + self.path + filename \
				+ ' --model_out ' + self.path + 'model' \
				+ ' --dimensionality ' + str(dimensionality)

		flag = subprocess.call(self.command_train, shell=True, stdout=open(self.path+'train.out', 'w'))
		if flag == 0:
			with open(self.path+'train.out', 'r') as f:
				out = ' '.join(list(f))
				end = re.findall('Objective function value for training: ([0-9.e+]+)', out)[0]
				print('extract percent: ', float(end))

		return flag


	def _transform(self, filename):
		self.command_test = ['sofia-kmeans',
								'--model_in', self.path + 'model',
								'--test_file', self.path + filename,
								'--cluster_mapping_out', self.path + 'mapping',
								'--cluster_mapping_type rbf_kernel',
								'--cluster_mapping_param', str(self.mapping_param),
								'--cluster_mapping_threshold', str(self.mapping_threshold)]
		self.command_test = ' '.join(self.command_test)

		flag = subprocess.call(self.command_test, shell=True, stdout=open(self.path+'test.out', 'w'))
		return flag


	def fit(self, X, y=None):
		X = self.whitener.fit_transform(X)
		dim = pd2svm(self.path + 'train.libsvm', X)
		flag = self._fit('train.libsvm', dim)
		if flag != 0:
			raise RuntimeError('training runtime error')


	def transform(self, X):
		X = self.whitener.transform(X)
		dim = pd2svm(self.path + 'test.libsvm', X)	
		flag = self._transform('test.libsvm')
		if flag != 0:
			raise RuntimeError('testing runtime error')

		mapping = load_svmlight_file(self.path + 'mapping')[0].toarray()
		print('sparcity: ', (mapping==0).sum() / mapping.shape[0] / mapping.shape[1])
		return mapping


	def fit_transform(self, X, y=None):
		X = self.whitener.fit_transform(X)
		dim = pd2svm(self.path + 'train.libsvm', X)	
		flag = self._fit('train.libsvm', dim)
		if flag != 0:
			raise RuntimeError('training runtime error')

		flag = self._transform('train.libsvm')
		if flag != 0:
			raise RuntimeError('testing runtime error')

		mapping = load_svmlight_file(self.path + 'mapping')[0].toarray()
		print('sparcity: ', (mapping==0).sum() / mapping.shape[0] / mapping.shape[1])
		return mapping

# ====================================================================
# xgboost wrapper

class Xgbooster(BaseEstimator, RegressorMixin):
	def __init__(self, n_trees=100, **kwargs):
		self.param = kwargs
		self.n_trees = n_trees

	def fit(self, X, y):
		dTrain = xgb.DMatrix(X, label=y)
		self.model = xgb.train(self.param, dTrain, self.n_trees)
		
	def predict(self, X):
		dTest = xgb.DMatrix(X)
		return self.model.predict(dTest)


# ====================================================================
# xgboost transformer

class XgbTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, n_trees=30, **kwargs):
		self.param = kwargs
		self.n_trees = n_trees

	def fit_transform(self, X, y):
		X, y = np.array(X), np.array(y)
		dTrain = xgb.DMatrix(X, label=y)
		self.model = xgb.train(self.param, dTrain, self.n_trees)
		code = self.model.predict(dTrain, pred_leaf=True)

		code = pd.DataFrame(code)
		for x in code.columns:
			code[x] = code[x].astype('category')

		code = pd.get_dummies(code)
		self.cols = code.columns
		return code.values

	def transform(self, X):
		X = np.array(X)
		dTrain = xgb.DMatrix(X)
		code = self.model.predict(dTrain, pred_leaf=True)

		code = pd.DataFrame(code)
		for x in code.columns:
			code[x] = code[x].astype('category')

		code = pd.get_dummies(code)
		missing = list(set(self.cols).difference(set(code.columns)))
		for x in missing:
			code[x] = 0
		return code[self.cols].values

# ====================================================================
# kfold transformer

class KFoldTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, base_estimator, K=2, **kwargs):
		self.K = K
		self.base_estimator = base_estimator
		self.estimators = []

		base_estimator_ = {'reg:RF': RandomForestRegressor,
						   'reg:ET': ExtraTreesRegressor,
						   'reg:XGB': Xgbooster,
						   'reg:KNN': KNeighborsRegressor,
						   'reg:L1': Lasso}

		for i in range(K):
			self.estimators.append(base_estimator_[base_estimator](**kwargs))
			

	def fit_transform(self, X, y):
		X, y = np.array(X), np.array(y)
		yhat = np.zeros(len(y))
		skf = KFold(len(y), n_folds=self.K)
		for i, (train, test) in enumerate(skf):
			self.estimators[i].fit(X[train, :], y[train])
			yhat[test] = self.estimators[i].predict(X[test, :])

		return np.expand_dims(yhat, axis=1)
			
	def transform(self, X):
		yhat = np.zeros(X.shape[0])
		for i in range(self.K):
			yhat += self.estimators[i].predict(X)

		return np.expand_dims(yhat, axis=1) / self.K


# ====================================================================
# K-fold Estimator 


class KFoldEstimator(BaseEstimator, TransformerMixin):
	def __init__(self, base_estimator, K=2, **kwargs):
		self.K = K
		self.base_estimator = base_estimator
		self.estimators = []

		base_estimator_ = {'reg:RF': RandomForestRegressor,
						   'reg:ET': ExtraTreesRegressor,
						   'reg:XGB': Xgbooster}

		for i in range(K):
			self.estimators.append(base_estimator_[base_estimator](**kwargs))
			

	def fit(self, X, y):
		skf = KFold(len(y), n_folds=self.K)
		for i, (train, test) in enumerate(skf):
			self.estimators[i].fit(X[train, :], y[train])


	def predict(self, X):
		yhat = np.zeros(X.shape[0])
		for i in range(self.K):
			yhat += self.estimators[i].predict(X)

		return yhat / self.K


# ====================================================================
# cross validation

def cross_validate(estimator, X, y, nfold=5, feval=gini.normalized_gini):
	X, y = np.array(X), np.array(y)
	scores = []
	kf = KFold(X.shape[0], nfold)
	for idx_train, idx_test in kf:
	    estimator.fit(X[idx_train, :], y[idx_train])
	    result = estimator.predict(X[idx_test, :])
	    scores.append(feval(y[idx_test], result))

	return np.mean(scores)


# ====================================================================
# ensemble regressor

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
				pred_ += rankdata(reg.predict(X))
			else:
				raise ValueError('Unknown pooling method')

		return pred_ / len(self.regressors)
