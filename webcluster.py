__authors__ = "Nicholas Leonard"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Nicholas Leonard"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"
__email__ = "leonardn@iro"

import numpy as np
import os
from itertools import izip
import theano
import theano.tensor as T
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import string_utils

class WebCluster(DenseDesignMatrix):
    """
    A multi-purpose Pylearn2 Dataset object using data crawled from the 
    Web and from the delicious.com web service in 2011.
    
    Each example represents a web document, i.e. a URI. 
    Inputs are the term frequencies of those documents, but only for the
    10000 terms found in the most documents. 
    For any document, they sum to 1.
    
    Targets are document clusters (classes) generated using the cosinus 
    similarity between each document's tags (and associated user 
    tagging frequency) followed by a custom similarity-graph clustering
    algorithm that is performed greedy-scale wise starting with the 
    cluster10s of documents, followed the cluster100s of cluster1000s.
    The 1000 cluster10s have a maximum of 10 documents per cluster.
    The 100 cluster100s have a maximum of 100 documents per cluster.
    The 10 cluster1000s have a maximum of 1000 documents per cluster.
    
    The corpus used here is itself a dense cluster formed around the
    following tags (with associated corpus-wide document count and 
    user tagging frequency):
    
    "wordpress";8507;599981
    "plugin";3067;92111
    "plugins";2628;80098
    "webdesign";2354;92517
    "blog";2244;64377
    "themes";1973;81755
    "theme";1804;49225
    "tutorial";1590;61799
    "design";1310;45107
    "wp";1220;9832
    "howto";1096;28114
    "blogging";931;24723
    "tips";899;32042
    "tutorials";867;36052
    "php";733;16127
    "templates";720;24386
    "cms";716;23778
    "free";676;19796
    "wordpress-plugins";653;8651
    "web";646;8761
    
    which gives you an idea of the kind of information 
    we can obtain from the delicious.com web service.
    """

    def __init__(self, which_set, data_path=None, 
                 term_range=None, target_type='cluster100'):
        """
        which_set: a string specifying which portion of the dataset
            to load. Valid values are 'train', 'valid' or 'test'
        data_path: a string specifying the directory containing the 
            webcluster data. If None (default), use environment 
            variable WEBCLUSTER_DATA_PATH.
        term_range: a tuple for taking only a slice of the available
            terms. Default is to use all 6275. For example, an input
            range of (10,2000) will truncate the 10 most frequent terms
            and the 6275-2000=4275 les frequent terms, whereby frequency
            we mean how many unique documents each term is in.
        target_type: the type of targets to use. Valid options are 
            'cluster[10,100,1000]'
        """
        self.__dict__.update(locals())
        del self.self
        
        self.corpus_terms = None
        self.doc_info = None
        
        print "loading WebCluster DDM. which_set =", self.which_set
        
        if self.data_path is None:
            self.data_path \
                = string_utils.preprocess('${WEBCLUSTER_DATA_PATH}')
        
        fname = os.path.join(self.data_path, which_set+'_doc_inputs.npy')
        X = np.load(fname)
        if self.term_range is not None:
            X = X[:,self.term_range[0]:self.term_range[1]]
            X = X/X.sum(1).reshape(X.shape[0],1)
        print X.sum(1).mean()
        
        fname = os.path.join(self.data_path, which_set+'_doc_targets.npy')
        # columns: 0:cluster10s, 1:cluster100s, 2:cluster1000s
        self.cluster_hierarchy = np.load(fname)
        
        y = None
        if self.target_type == 'cluster10':
            y = self.cluster_hierarchy[:,0]
        elif self.target_type == 'cluster100':
            y = self.cluster_hierarchy[:,1]
        elif self.target_type == 'cluster1000':
            y = self.cluster_hierarchy[:,2]
        elif self.target_type is None:
            pass
        else:
            raise NotImplementedError()
        
        DenseDesignMatrix.__init__(self, X=X, y=y)
        
        print "... WebCluster ddm loaded"
        
    def get_input_shape(self):
        return self.X.shape
        
    def get_corpus_terms(self):
        if self.corpus_terms is None:
            fname = os.path.join(self.data_path, 'corpus_terms.npy')
            self.corpus_terms = np.load(fname)
            if self.term_range is not None:
                self.corpus_terms = self.corpus_terms[ \
                                self.term_range[0]:self.term_range[1]]
                                
        return self.corpus_terms
        
    def get_doc_info(self):
        if self.doc_info is None:
            fname = os.path.join(self.data_path, self.which_set+'_doc_info.npy')
        self.doc_info = np.load(fname)
        return self.doc_info

    def get_target_shape(self):
        return self.y.shape
        
if __name__ == '__main__':
    web_ddm = WebCluster(which_set='train')
    web_ddm2 = WebCluster(which_set='train',term_range=(10,2000))
    #import pdb; pdb.set_trace()
