__authors__ = "Nicholas Leonard, Yann Dauphin"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Nicholas Leonard", "Yann Dauphin"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"
__email__ = "leonardn@iro"

import numpy as np
import os
from itertools import izip
import theano
import theano.tensor as T
from sparse_design_matrix import SparseDesignMatrix
from pylearn2.utils import string_utils
import scipy.sparse as ssp

class Newsgroups20(SparseDesignMatrix):
    """
    A multi-purpose Pylearn2 Dataset for the 20 newsgroups dataset
    http://qwone.com/~jason/20Newsgroups/
    """

    def __init__(self, which_set, data_path=None, valid_ratio=0.2,
                 sum_to_one=True, one_hot=True):
        """
        which_set: a string specifying which portion of the dataset
            to load. Valid values are 'train', 'valid' or 'test'
        data_path: a string specifying the directory containing the 
            20 newsgroups data. If None (default), use environment 
            variable NEWSGROUPS_DATA_PATH.
        valid_ratio: ratio of the non-test examples that will be used 
            for the 'valid' set. Also means that 1.0-valid_raito is the
            ratio of the non-text examples that will be used for the 
            'train' set.
        sum_to_one: when True (default), makes the term frequencies of
            each example sum to one. This shouldn't be necessary for 
            networks that use rectified linear hidden units with a 
            softmax at the output layer since it is scale invariant and
            allows the network to learn something from the size of each
            document.
        one_host: when True (default), maps the loaded targets into
            arrays of lenght 20 with one 1.
        """
        self.__dict__.update(locals())
        del self.self
        
        print "loading Newsgroups20 DDM. which_set =", self.which_set
        
        if self.data_path is None:
            self.data_path \
                = string_utils.preprocess('${NEWSGROUPS_DATA_PATH}')
        
        if which_set == 'valid':
            which_set = 'train'
        fname = os.path.join(self.data_path, which_set+'_y.npy')
        y = np.load(fname)
        fname = os.path.join(self.data_path, which_set+'_x.npy')
        X = np.load(fname).item()
        #import pdb; pdb.set_trace()
        if self.which_set == 'train':
            # Load train set
            print X.shape
            idx = int((1-self.valid_ratio)*X.shape[0])
            X = X[:idx,:]
            y = y[:idx]
        elif self.which_set == 'valid':
            # Load valid set
            idx = int((1-self.valid_ratio)*X.shape[0])
            X = X[idx:,:]
            y = y[idx:]
            
        
        if self.one_hot:
            one_hot = np.zeros((y.shape[0],20),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot
            
        if self.sum_to_one:
            d = ssp.lil_matrix((X.shape[0],X.shape[0]))
            d.setdiag((1./np.asarray(X.sum(1))).reshape(X.shape[0]))
            X = d*X
        
        SparseDesignMatrix.__init__(self, X=X, y=y)
        
        print "... Newsgroup20 ddm loaded"
        
    def get_input_shape(self):
        return self.X.shape

    def get_target_shape(self):
        return self.y.shape
        
def parse_dataset(data_path='dataset/'):
    import numpy
    from sklearn.feature_extraction.text import CountVectorizer

    numpy.random.seed(0xbeef)

    train = open(data_path+"20ng-train-stemmed.txt").read().splitlines()
    test = open(data_path+"20ng-test-stemmed.txt").read().splitlines()

    train = map(str.split, train)
    train_x = map(lambda x: " ".join(x[1:]), train)
    train_y = map(lambda x: x[0], train)

    test = map(str.split, test)
    test_x = map(lambda x: " ".join(x[1:]), test)
    test_y = map(lambda x: x[0], test)

    vec = CountVectorizer(min_df=0.0)
    vec.fit(train_x)

    train_x = vec.transform(train_x).tocsr()
    test_x = vec.transform(test_x).tocsr()

    train_x.data = numpy.asarray(train_x.data, "float32")
    test_x.data = numpy.asarray(test_x.data, "float32")

    labels = list(set(train_y))
    labels = dict(zip(labels, range(len(labels))))

    train_y = numpy.asarray([labels[l] for l in train_y], 'int32')
    test_y = numpy.asarray([labels[l] for l in test_y], 'int32')

    inds = range(train_x.shape[0])
    numpy.random.shuffle(inds)
    train_x = train_x[inds]
    train_y = train_y[inds]

    numpy.save(data_path+"train_x.npy", train_x)
    numpy.save(data_path+"train_y.npy", train_y)
    numpy.save(data_path+"test_x.npy", test_x)
    numpy.save(data_path+"test_y.npy", test_y)
    #import pdb; pdb.set_trace()
        
if __name__ == '__main__':
    #parse_dataset()
    train_sdm = Newsgroups20(which_set='train')
    valid_sdm = Newsgroups20(which_set='valid', sum_to_one=True)
    test_sdm = Newsgroups20(which_set='test')
