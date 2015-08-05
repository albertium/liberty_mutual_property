import os
import shutil
import re
import numpy as np
import pandas as pd
import subprocess
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.datasets import load_svmlight_file
from numpy.linalg import eig

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

# =============================================================================================================

class Whitener(BaseEstimator, ClusterMixin, TransformerMixin):
	def __init__(self, var_e=0.01, eig_e=1E-5):
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


# =============================================================================================================

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
