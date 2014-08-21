#!/usr/bin/python
# this is the example script to use xgboost to train 
import inspect
import os
import sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold
# add path of xgboost python module
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../python")

sys.path.append(code_path)

import xgboost as xgb

test_size = 550000

def get_params(sum_wneg, sum_wpos):
    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    param['objective'] = 'binary:logitraw'
    # scale weight of positive examples
    param['scale_pos_weight'] = sum_wneg/sum_wpos
    param['bst:eta'] = 0.1
    param['bst:max_depth'] = 9
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['nthread'] = 16
    
    # you can directly throw param in, though we want to watch multiple metrics here 
    return list(param.items())+[('eval_metric', 'ams@0.15')]

# path to where the data lies
dpath = 'data'

# load in training data, directly use numpy
dtrain = np.loadtxt( dpath+'/training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
print ('finish loading from csv ')

dlabel = dtrain[:,32]

dindex = 1
for train, test in StratifiedKFold(dlabel, 5):
    tlabel  = dlabel[test]
    tdata   = dtrain[test,1:31]
    tweight = dtrain[test,31]
    
    label   = dlabel[train]
    data    = dtrain[train,1:31]
    weight  = dtrain[train,31] * float(len(tlabel)) / len(label)

    sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
    sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )
    
    # print weight statistics 
    print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))
    
    # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
    xgmat = xgb.DMatrix( data, label=label, missing = -999.0, weight=weight )
    txgmat = xgb.DMatrix( tdata, label=tlabel, missing = -999.0, weight=tweight )

    watchlist = [ (xgmat,'train'), (txgmat,'test') ]
    # boost 120 tres
    num_round = 120
    print ('loading data end, start to boost trees')
    bst = xgb.train( get_params(sum_wneg, sum_wpos), xgmat, num_round, watchlist );
    # save out model
    bst.save_model('higgs.model.%d' % dindex)
    dindex += 1

print ('finish training')
