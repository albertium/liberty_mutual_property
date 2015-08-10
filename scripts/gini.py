import numpy as np
from sklearn.metrics import make_scorer

def gini_coef(target, pred):
    """target, input: ndarray"""
    target, pred = np.array(target), np.array(pred)
    assert len(target) == len(pred)
    idx = np.lexsort((target, pred))
    cum = np.cumsum(target[idx])
    cum = cum / float(cum[-1])
    return ((1+len(target))/2.0 - np.sum(cum)) / len(target)


def normalized_gini(target, pred):
	return gini_coef(target, pred) / gini_coef(target, target)	


def xgb_gini(pred, dtrain):
	return 'gini', normalized_gini(dtrain.get_label(), pred)


normalized_gini_score = make_scorer(normalized_gini, greater_is_better=True)
