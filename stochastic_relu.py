# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:56:19 2013

@author: Nicholas Léonard
"""

import time, sys

from pylearn2.utils import serial
from itertools import izip
from pylearn2.utils import safe_zip
from collections import OrderedDict
from pylearn2.utils import safe_union

import numpy as np
import theano.sparse as S

from theano.gof.op import get_debug_values
from theano.printing import Print
from theano import function
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T
import theano

from pylearn2.linear.matrixmul import MatrixMul

from pylearn2.models.model import Model

from pylearn2.utils import sharedX

from pylearn2.costs.cost import Cost
from pylearn2.models.mlp import MLP, Softmax, Layer, Linear, RectifiedLinear
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
from hps3 import MLPCost


class Stochastic2Cost(MLPCost):
    def get_gradients(self, model, X, Y=None, ** kwargs):
        """
        model: a pylearn2 Model instance
        X: a batch in model.get_input_space()
        Y: a batch in model.get_output_space()

        returns: gradients, updates
            gradients:
                a dictionary mapping from the model's parameters
                         to their gradients
                The default implementation is to compute the gradients
                using T.grad applied to the value returned by __call__.
                However, subclasses may return other values for the gradient.
                For example, an intractable cost may return a sampling-based
                approximation to its gradient.
            updates:
                a dictionary mapping shared variables to updates that must
                be applied to them each time these gradients are computed.
                This is to facilitate computation of sampling-based approximate
                gradients.
                The parameters should never appear in the updates dictionary.
                This would imply that computing their gradient changes
                their value, thus making the gradient value outdated.
        """

        try:
            if Y is None:
                cost = self(model=model, X=X, **kwargs)
            else:
                cost = self(model=model, X=X, Y=Y, **kwargs)
        except TypeError,e:
            # If anybody knows how to add type(seslf) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling "+str(type(self))+".__call__"
            print str(type(self))
            print e.message
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self))+" represents an intractable "
                    " cost and does not provide a gradient approximation scheme.")

        params = list(model.get_params())
        
        layers = model.layers
        
        '''Get layer costs'''
        for layer in layers:
            if hasattr(layer,'get_cost'):
                cost += layer.get_cost()

        grads = T.grad(cost, params, disconnected_inputs = 'raise')

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates        

class SReLU(Linear):
    """
    A layer for semi-hard stochastic neurons, i.e. stochastic rectified
    linear units (SReLU).
    """

    def __init__(self,
                 dim,
                 stdev,
                 layer_name,
                 irange = None,
                 istdev = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 mask_weights = None,
                 max_row_norm = None,
                 max_col_norm = None,
                 softmax_columns = False,
                 copy_input = 0):
        '''
        params
        ------
        dim: 
            number of units on output layer
        stdev:
            standard deviation of gaussian noise.
        
        '''
                     
        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.dim,)) + init_bias, name = layer_name + '_b')

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])

    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict()

        mx = state.max(axis=0)
        mean = state.mean(axis=0)
        mn = state.min(axis=0)

        rval['max_x_max_u'] = mx.max()
        rval['max_x_mean_u'] = mx.mean()
        rval['max_x_min_u'] = mx.min()

        rval['mean_x_max_u'] = mean.max()
        rval['mean_x_mean_u'] = mean.mean()
        rval['mean_x_min_u'] = mean.min()

        rval['min_x_max_u'] = mn.max()
        rval['min_x_mean_u'] = mn.mean()
        rval['min_x_min_u'] = mn.min()

        return rval

    def _linear_part(self, state_below):
        # TODO: Refactor More Better(tm)
        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        if self.softmax_columns:
            W, = self.transformer.get_params()
            W = W.T
            W = T.nnet.softmax(W)
            W = W.T
            z = T.dot(state_below, W) + self.b
        else:
            z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'
        if self.copy_input:
            z = T.concatenate((z, state_below), axis=1)
        return z


    def fprop(self, state_below):
        # TODO: Refactor More Better(tm)
        self.p = self._linear_part(state_below)
        
        # sample from bernouili probs to generate a mask
        rng = MRG_RandomStreams(self.mlp.rng.randint(2**15))
        m = rng.normal(size=self.p.shape, avg=0.0, std=self.stdev, 
                dtype=theano.config.floatX)
        
        return max(0, m+self.p)
        
    def test_fprop(self, state_below):
        p = self._linear_part(state_below)
        return max(0, p)
