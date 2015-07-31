import numpy as np

def gini_coef(target, pred):
    """target, input: ndarray"""
    assert len(target) == len(pred)
    idx = np.lexsort((target, pred))
    cum = np.cumsum(target[idx])
    cum = cum / float(cum[-1])
    return ((1+len(target))/2.0 - np.sum(cum)) / len(target)

def normalized_gini(target, pred):
	return gini_coef(target, pred) / gini_coef(target, target)	

def xgb_gini(pred, dtrain):
	return 'gini', normalized_gini(dtrain.get_label(), pred)
