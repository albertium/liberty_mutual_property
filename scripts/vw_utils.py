import pandas as pd
import numpy as np
import subprocess
import re
import os, shutil
import gini

def pd2vw(data, X, out, label=None):
    tmp = data.copy()
    if label:
    	ordering = [label]
    else:
    	ordering = []

    for vtype, namespaces in X.items():
        for namespace, labels in namespaces.items():
            ordering.append(namespace)
            tmp[namespace] = '|' + namespace        

            for lable in labels:
                ordering.append(lable)
                if vtype == 'cat':
                    tmp[lable] = lable+'_'+tmp[lable].map(str)
                elif vtype == 'num':
                    tmp[lable] = lable+':'+tmp[lable].map(str)
                else:
                    raise ValueError('Undefined processing type')

    tmp[ordering].to_csv('../tmp_data/{}'.format(out), index=False, header=False, sep=' ')



def vwRegress(data, out, passes=4, holdout=10, bit=25, l1=0, l2=0, q=None, learning_rate=0.5):
    path = '../tmp/' + out
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    # deal with q term
    if q:
    	q = ' '.join(['-q '+x for x in q.split(' ')])
    else:
    	q = ''
        
    command = "vw -d ../tmp_data/{0} --passes {1} -k -c --holdout_period {2} -b {3} -f {4}/model " + \
              "--readable_model {4}/model.r " + \
              "--loss_function squared --l1 {5} --l2 {6} -l {7} {8}"
    command = command.format(data, passes, holdout, bit, path, l1, l2, learning_rate, q)
    flag = subprocess.call(command, shell=True, stderr=open('{}/log'.format(path), 'w'))
    
    loss = ''
    if flag == 0:
        with open('{}/log'.format(path), 'r') as f:
            loss = re.findall('average loss = ([0-9.]+) h', (' '.join(list(f))))[0]
            
    return({'flag': flag,
            'loss': float(loss),
            'command': command})



def vwPredict(data, model, metric=False):
    path = '../tmp/' + model
    if not os.path.exists(path):
        raise ValueError('model path not exists')
        
    command = "vw -d ../tmp_data/{0} -t -i {1}/model -p {1}/out".format(data, path)
    flag = subprocess.call(command, shell=True, stderr=open('{}/log.out'.format(path), 'w'))
    
    score = 0
    if metric:
        y = []
        with open('../tmp_data/train.vw', 'r') as f:
            for line in f:
                y.append(float(re.findall('^([0-9.]+) ', line)[0]))
                
        yhat = np.loadtxt('../tmp/linear1/out')
        score = gini.normalized_gini(np.array(y), yhat)
    
    return({'flag': flag,
            'gini': score})
