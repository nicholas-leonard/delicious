__author__ = 'Vincent Archambault-Bouffard'
__credits__ = ['Ian Goodfellow', 'Vincent Archambault-Bouffard']

import sys
import numpy as np
import csv
from theano import function
from theano import tensor as T

from stochastic_hps import *


def usage():
    print """usage: python analyse.py model.pkl dataset_id
Where model.pkl contains a trained pylearn2.models.mlp.MLP object.
The script will make submission.csv, which you may then upload to the
kaggle site."""


if len(sys.argv) != 3:
    usage()
    print "(You used the wrong # of arguments)"
    quit(-1)

_, model_path, dataset_id = sys.argv

from pylearn2.utils import serial

try:
    model = serial.load(model_path)
except Exception, e:
    usage()
    print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
    print e

from pylearn2.config import yaml_parse

hps = StochasticHPS('analyse',-1,None)
row =  hps.db.executeSQL("""
SELECT preprocess_array,train_ddm_id,valid_ddm_id,test_ddm_id
FROM hps3.dataset
WHERE dataset_id = %s
""", (dataset_id,), hps.db.FETCH_ONE)
if not row or row is None:
    assert False
(preprocess_array,train_ddm_id,valid_ddm_id,test_ddm_id)  = row
# preprocessing
hps.load_preprocessor(preprocess_array)
# dense design matrices
hps.train_ddm = hps.get_ddm(train_ddm_id)
hps.valid_ddm = hps.get_ddm(valid_ddm_id)
hps.test_ddm = hps.get_ddm(test_ddm_id)
hps.apply_preprocess()
dataset = hps.test_ddm

# use smallish batches to avoid running out of memory
batch_size = 10
print hps.test_ddm.X.shape
model.set_batch_size(batch_size)
# dataset must be multiple of batch size of some batches will have
# different sizes. theano convolution requires a hard-coded batch size
#assert dataset.X.shape[0] % batch_size == 0

l = model.layers[0]
#import pdb; pdb.set_trace()
l.beta_mean = l.beta_dist.get_value().mean()

X = model.get_input_space().make_batch_theano()
target = T.matrix('target') 
# (batch_size, 30, 98)
H = model.layers[0].test_fprop(X)
Y = model.layers[1].fprop(H)
MCA = T.mean(T.cast(T.neq(T.argmax(Y, axis=1), 
                       T.argmax(target, axis=1)), dtype='int32'),
                       dtype=theano.config.floatX)
                       
Y2 = model.fprop(X)
MCA2 = T.mean(T.cast(T.neq(T.argmax(Y2, axis=1), 
                       T.argmax(target, axis=1)), dtype='int32'),
                       dtype=theano.config.floatX)
                       
sparsity = model.layers[0].m_mean.mean()

f = function([X, target], [MCA, sparsity, MCA2])

y = []
y2 = []
s = []

for imgIdx in xrange(dataset.X.shape[0] / batch_size):
    x_arg = dataset.X[imgIdx * batch_size:(imgIdx + 1) * batch_size, :]
    y_arg = dataset.y[imgIdx * batch_size:(imgIdx + 1) * batch_size, :]
    if X.ndim > 2:
        x_arg = dataset.get_topological_view(x_arg)
    r = f(x_arg.astype(X.dtype), y_arg)
    y.append(r[0])
    y2.append(r[2])
    s.append(r[1])
    

mca = np.asarray(y).mean()
mca2 = np.asarray(y2).mean()
s = np.asarray(s).mean()
print mca, s, mca2

