import os
import shutil
import re
import numpy as np
import pandas as pd
import subprocess
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

class sofia_kmeans(BaseEstimator, ClusterMixin, TransformerMixin):
	"""
	wrapper for sofia kmean via harddrive
	"""
	def __init__(self, name='sofia_0', n_clusters=10, batch_size=100, iterations=100):
		self.command = 'sofia-kmeans'
		self.command += ' --k ' + str(n_clusters) \
					 +  ' --init_type optimized_kmeans_pp' \
					 +  ' --opt_type mini_batch_kmeans' \
					 +  ' --mini_batch_size ' + str(batch_size) \
					 +  ' --iterations ' + str(iterations) \
					 +  ' --objective_after_init' \
					 +  ' --objective_after_training'

		self.path = '../tmp/' + name + '/'
		if not os.path.exists(self.path):
			os.makedirs(self.path)
		else:
			shutil.rmtree(self.path)
			os.makedirs(self.path)


	def _fit(self, filename):
		self.command_train = self.command \
				+ ' --training_file ' + self.path + filename \
				+ ' --model_out ' + self.path + 'model' \
				+ ' --dimensionality ' + str(data.shape[1])
		print(self.command_train)
		flag = subprocess.call(self.command_train, shell=True, stdout=open(self.path+'out', 'w'))
		if flag == 0:
			with open(self.path+'out', 'r') as f:
				out = ' '.join(list(f))
				start = re.findall('Objective function value for initialization: ([0-9.e+]+)', out)[0]
				end = re.findall('Objective function value for training: ([0-9.e+]+)', out)[0]
				print('extract percent: ', float(end))

		return flag


	def fit(self, data):
		_pd2svm(self.path + 'train.libsvm', data)
		if self._fit(self.path + 'train.libsvm') != 0:
			raise RuntimeError



def _pd2svm(filename, data):
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