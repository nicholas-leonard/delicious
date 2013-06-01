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
from pylearn2.models.mlp import MLP, Softmax, Layer, Linear
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
 
class Stochastic1Cost(Cost):
    supervised = True
    def __init__(self, cost_type='default'):
        self.__dict__.update(locals())
        del self.self
        self.use_dropout = False
    
    def setup_dropout(self, default_input_include_prob=.5, 
                        input_scales=None, input_include_probs=None, 
                        default_input_scale=2.):
        """
        During training, each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.
        """

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self
        
        self.use_dropout=True

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
            
        
                   
        layers = model.layers
        
        '''Get layer costs'''
        '''for layer in layers:
            cost += layer.get_cost()'''
            
        
        params = [self.z]
        grads = T.grad(cost, params, disconnected_inputs = 'raise', 
                        consider_constant=model.get_params())

        known_grads = OrderedDict(izip(params, grads))
        rupdates = OrderedDict()
        rgradients = OrderedDict()
        
        '''In reverse order, get gradients one layer at a time.'''
        for layer in reversed(layers):
            gradients, updates \
                = layer.get_gradients(known_grads.copy(), self.loss)
            known_grads.update(gradients)
            rupdates.update(updates)
            rgradients.update(gradients)
            
        '''print len(rgradients), len(rupdates), len(known_grads)
        
        for param in model.get_params():
            print param.name
            
        print 'grads'
        for (param, grad) in rgradients.iteritems():
            print param.name, grad'''

        return rgradients, rupdates
        
    def __call__(self, model, X, Y, ** kwargs):
        if self.use_dropout:
            Y_hat = model.dropout_fprop(X, default_input_include_prob=self.default_input_include_prob,
                    input_include_probs=self.input_include_probs, default_input_scale=self.default_input_scale,
                    input_scales=self.input_scales
                    )
        else:
            Y_hat = model.fprop(X)
        
        if self.cost_type == 'default':
            # nll
            assert hasattr(Y_hat, 'owner')
            owner = Y_hat.owner
            assert owner is not None
            op = owner.op
            assert isinstance(op, T.nnet.Softmax)
            z ,= owner.inputs
            self.z = z
            assert z.ndim == 2

            z = z - z.max(axis=1).dimshuffle(0, 'x')
            log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
            # we use sum and not mean because this is really one variable per row
            log_prob_of = (Y * log_prob).sum(axis=1)
            assert log_prob_of.ndim == 1

            rval = log_prob_of.mean()

            cost = - rval
            self.loss = (-Y * T.log(Y_hat)).sum(axis=1)
        else:
            raise NotImplementedError()
        return cost
        
    def get_test_cost(self, model, X, Y):
        state_below = X
        for layer in model.layers:
            if hasattr(layer, 'test_fprop'):
                state_below = layer.test_fprop(state_below)
            else:
                state_below = layer.fprop(state_below)
        y = state_below
        MCE = T.mean(T.cast(T.neq(T.argmax(y, axis=1), 
                       T.argmax(Y, axis=1)), dtype='int32'),
                       dtype=config.floatX)
        return MCE
        
      
   
class Group1(Layer):
    """
    We use the biased low-variance estimator to estimate gradients of 
    stochastic neurons of the gater. Each such gater neuron represents a 
    group of neurons (an expert) in the main part
    """

    def __init__(self,
                 gater_dim,
                 hidden_dim,
                 expert_dim,
                 layer_name,
                 hidden_activation = 'tanh',
                 expert_activation = None,
                 derive_sigmoid = True,
                 sparsity_target = 0.1,
                 sparsity_cost_coeff = 1.0,
                 irange = [None,None,None],
                 istdev = [None,None,None],
                 sparse_init = [None,None,None],
                 sparse_stdev = [1.,1.,1.],
                 init_bias = [0.,0.,0.],
                 W_lr_scale = [None,None,None],
                 b_lr_scale = [None,None,None],
                 max_col_norm = [None,None,None],
                 weight_decay_coeff = [None,None,None]):
        '''
        params
        ------
        dim: 
            number of units on output layer
        hidden_dim: 
            number of units on hidden layer of non-linear part
        hidden_activation:
            activation function used on hidden layer of non-linear part
        sparsity_target:
            target sparsity of the output layer.
        sparsity_cost_coeff:
            coefficient of the sparsity constraint when summing costs
        weight_decay_coeff:
            coefficients of L2 weight decay when summing costs
        other:
            in the lists of params, the first index is for the linear 
            part, while the second and third indices are for the first 
            and second layer of the non-linear part, respectively
        
        '''
                     
        self.__dict__.update(locals())
        del self.self
        
        assert expert_activation is None

    def get_lr_scalers(self):
        rval = OrderedDict()

        for i in range(3):
            if self.W_lr_scale[i] is not None:
                rval[self.W[i]] = self.W_lr_scale[i]

            if self.b_lr_scale[i] is not None:
                rval[self.b[i]] = self.b_lr_scale[i]

        return rval

    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        # units per expert times number of experts:
        self.dim = self.expert_dim*self.gater_dim
        self.output_space = VectorSpace(self.dim)

        self.input_dims = [self.input_dim, self.hidden_dim]
        self.output_dims = [self.hidden_dim, self.dim]
    
        self.W = [None,None]
        self.b = [None,None]
        
        for i in range(2):
            self._init_inner_layer(i)
            
        self.W = [None] + self.W
        self.b = [None] + self.b
        
        self._init_expert_layer()
        
        self.stoch_grad = sharedX(0)
        self.kl_grad = sharedX(0)
        self.linear_grad = sharedX(0)
        
    def _init_expert_layer(self, idx=0):
        rng = self.mlp.rng
        if self.irange[idx] is not None:
            assert self.istdev[idx] is None
            assert self.sparse_init[idx] is None
            W = rng.uniform(-self.irange[idx], self.irange[idx],
                    (self.input_dim, self.gater_dim, self.expert_dim))
        elif self.istdev[idx] is not None:
            assert self.sparse_init[idx] is None
            W = rng.randn(self.input_dim, self.gater_dim,
                            self.expert_dim) * self.istdev[idx]
        else:
            assert self.sparse_init[idx] is not None
            raise NotImplementedError()
            W = np.zeros((self.input_dim, self.gater_dim, self.expert_dim))
            for i in xrange(self.output_dims[idx]):
                assert self.sparse_init[idx] <= self.input_dims[idx]
                for j in xrange(self.sparse_init[idx]):
                    idx2 = rng.randint(0, self.input_dims[idx])
                    while W[idx2, i] != 0:
                        idx2 = rng.randint(0, self.input_dims[idx])
                    W[idx2, i] = rng.randn()
            W *= self.sparse_stdev[idx]

        W = sharedX(W)
        W.name = self.layer_name + '_W' + str(idx)
        
        b = sharedX( np.zeros((self.gater_dims,self.expert_dim)) \
                + self.init_bias[idx], \
                name = self.layer_name + '_b' + str(idx))

        self.W[idx] = W
        self.b[idx] = b
        
    def _init_inner_layer(self, idx):
        rng = self.mlp.rng
        if self.irange[idx] is not None:
            assert self.istdev[idx] is None
            assert self.sparse_init[idx] is None
            W = rng.uniform(-self.irange[idx], self.irange[idx],
                        (self.input_dims[idx], self.output_dims[idx]))
        elif self.istdev[idx] is not None:
            assert self.sparse_init[idx] is None
            W = rng.randn(self.input_dims[idx], self.output_dims[idx]) \
                    * self.istdev[idx]
        else:
            assert self.sparse_init[idx] is not None
            W = np.zeros((self.input_dims[idx], self.output_dims[idx]))
            for i in xrange(self.output_dims[idx]):
                assert self.sparse_init[idx] <= self.input_dims[idx]
                for j in xrange(self.sparse_init[idx]):
                    idx2 = rng.randint(0, self.input_dims[idx])
                    while W[idx2, i] != 0:
                        idx2 = rng.randint(0, self.input_dims[idx])
                    W[idx2, i] = rng.randn()
            W *= self.sparse_stdev[idx]

        W = sharedX(W)
        W.name = self.layer_name + '_W' + str(idx)
        
        b = sharedX( np.zeros((self.output_dims[idx],)) \
                + self.init_bias[idx], \
                name = self.layer_name + '_b' + str(idx))

        self.W[idx] = W
        self.b[idx] = b


    def censor_updates(self, updates):
        for idx in range(3):
            if self.max_col_norm[idx] is not None:
                W = self.W[idx]
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm[idx])
                    updates[W] = updated_W * desired_norms / (1e-7 + col_norms)


    def get_params(self):
        rval = [self.W[0], self.W[1], self.W[2], self.b[0], self.b[1], self.b[2]]
        return rval

    def get_weights(self):
        rval = []
        for i in range(3):
            W = self.W[i].get_value()
            raise NotImplementedError()
            # should code means of seeing weights in groups
            rval.append(W.get_value())
            
        return rval

    def set_weights(self, weights):
        for i in range(3):
            W = self.W[i]
            W.set_value(weights[i])

    def set_biases(self, biases):
        for i in range(3):
            self.b[i].set_value(biases[i])

    def get_biases(self):
        rval = []
        for i in range(3):
            rval.append(self.b[i].get_value())
        return rval

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_topo(self):
        raise NotImplementedError()
        
    def get_monitoring_channels(self):
        rval = OrderedDict()
        rval['stoch_grad'] = self.stoch_grad
        rval['kl_grad'] = self.kl_grad
        rval['linear_grad'] = self.linear_grad
        
        for i in range(3):
            sq_W = T.sqr(self.W[i])

            row_norms = T.sqrt(sq_W.sum(axis=1))
            col_norms = T.sqrt(sq_W.sum(axis=0))
            
            rval['row_norms_max'+str(i)] = row_norms.max()
            rval['col_norms_max'+str(i)] = col_norms.max()
        
        return rval
        
    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict()
        # sparisty of outputs:
        rval['mean_output_sparsity'] = self.m_mean.mean()
        # proportion of sigmoids that have prob > 0.5
        # good when equal to sparsity
        floatX = theano.config.floatX
        rval['mean_sparsity_prop'] \
            = T.cast(T.gt(self.m_mean, 0.5),floatX).mean()
        # same as above but for intermediate thresholds:
        rval['mean_sparsity_prop0.2'] \
            = T.cast(T.gt(self.m_mean, 0.2),floatX).mean()
        rval['mean_sparsity_prop0.3'] \
            = T.cast(T.gt(self.m_mean, 0.3),floatX).mean()
        rval['mean_sparsity_prop0.4'] \
            = T.cast(T.gt(self.m_mean, 0.4),floatX).mean()    
        # or just plain standard deviation (less is bad):
        rval['output_stdev'] = self.m_mean.std()
        # stdev of unit stdevs (more is bad)
        rval['output_meta_stdev'] = self.m_mean.std(axis=0).std()
        # max and min proportion of these probs per unit
        prop_per_unit = T.cast(T.gt(self.m_mean, 0.5),floatX).mean(0)
        # if this is high, it means a unit is likely always active (bad)
        rval['max_unit_sparsity_prop'] = prop_per_unit.max()
        rval['min_unit_sparsity_prop'] = prop_per_unit.min()
        # in both cases, high means units are popular (bad)
        # proportion of units with p>0.5 more than 50% of time:
        rval['mean_unit_sparsity_meta_prop'] \
            = T.cast(T.gt(prop_per_unit,0.5),floatX).mean()
        # proportion of units with p>0.5 more than 75% of time:
        rval['mean_unit_sparsity_meta_prop2'] \
            = T.cast(T.gt(prop_per_unit,0.75),floatX).mean()
        
        return rval

    def fprop(self, state_below, threshold=None, stochastic=True):
        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)
        
        self.x = state_below
        
        # linear part
        if isinstance(self.x, S.SparseVariable):
            raise NotImplementedError()
            z = S.dot(self.x,self.W[0]) + self.b[0]
        else:
            # w : (input_dim,gater_dim,expert_dim)
            # b : (gater_dim,expert_dim)
            # x : (batch_size,input_dim)
            # z : (batch_size,gater_dim,expert_dim)
            z = T.tensordot(self.x,self.W[0],axes=[[1],[0]]) + self.b[0].dimshuffle('x',0,1)
        
        '''self.output_activation = None
        # activate hidden units of non-linear part
        if self.output_activation is None:
            self.z = z
        elif self.hidden_activation == 'tanh':
            self.z = T.tanh(z)
        elif self.output_activation == 'sigmoid':
            self.z = T.nnet.sigmoid(z)
        elif self.output_activation == 'softmax':
            self.z = T.nnet.softmax(z)
        elif self.output_activation == 'rectifiedlinear':
            self.z = T.maximum(0, z)
        else:
            raise NotImplementedError()'''
        
        # first layer non-linear part
        if isinstance(self.x, S.SparseVariable):
            h = S.dot(self.x,self.W[1]) + self.b[1]
        else:
            h = T.dot(self.x,self.W[1]) + self.b[1]
        
        # activate hidden units of non-linear part
        if self.hidden_activation is None:
            self.h = h
        elif self.hidden_activation == 'tanh':
            self.h = T.tanh(h)
        elif self.hidden_activation == 'sigmoid':
            self.h = T.nnet.sigmoid(h)
        elif self.hidden_activation == 'softmax':
            self.h = T.nnet.softmax(h)
        elif self.hidden_activation == 'rectifiedlinear':
            self.h = T.maximum(0, h)
        else:
            raise NotImplementedError()
        
        # second layer non-linear part
        self.a = T.dot(self.h,self.W[2]) + self.b[2]
        
        # activate non-linear part to get bernouilli probabilities
        self.m_mean = T.nnet.sigmoid(self.a)
        
        if threshold is None:
            if stochastic:
                # sample from bernouili probs to generate a mask
                rng = MRG_RandomStreams(self.mlp.rng.randint(2**15))
                self.m = rng.binomial(size = self.m_mean.shape, n = 1, 
                        p = self.m_mean, dtype=self.m_mean.type.dtype)
            else:
                self.m = self.m_mean
        else:
            # deterministic mask:
            self.m = T.cast(T.gt(self.m_mean, threshold), \
                                        theano.config.floatX)
           
        
        # mask output of experts part with samples from gater part
        # m: (batch_size, gater_dim)
        # z: (batch_size, gater_dim, expert_dim)
        # p: (batch_size, gater_dim*expert_dim)
        self.p = (self.m.dimshuffle(0,1,'x') * self.z).flatten(2)
        
        if self.layer_name is not None:
            self.z.name = self.layer_name + '_z'
            self.h.name = self.layer_name + '_h'
            self.a.name = self.layer_name + '_a'
            self.m_mean.name = self.layer_name + '_m_mean'
            self.m.name = self.layer_name + '_m'
            self.p.name = self.layer_name + '_p'
        
        return self.p
        
    def test_fprop(self, state_below, threshold=None, stochastic=True):
        return self.fprop(state_below, threshold, stochastic)
        
    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y - Y_hat)
        
    def get_gradients(self, known_grads, loss):
        '''
        Computes gradients and updates for this layer given the known
        gradients of the upper layers, and the vector of losses for the
        batch.
        '''
        updates = OrderedDict()
        
        cost = self.get_kl_divergence() + self.get_weight_decay()
        # gradient of experts.
        params = [self.W[0], self.b[0]]
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads, 
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='raise')
        cost_grads = T.grad(cost=cost, wrt=params,
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='ignore')
                    
        updates[self.linear_grad] = T.abs_(grads[0]).mean()
        
        for i in range(len(grads)):
            grads[i] += cost_grads[i]
            
        gradients = OrderedDict(izip(params, grads))
        
        # gradients of non-linear part:
        ## start by getting gradients at binary mask:
        params = [self.m]
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads, 
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='raise')
        print "grads at bin", grads
        
        # estimate gradient at simoid input using above:
        grad_m = grads[0]
        if self.derive_sigmoid:
            # multiplying by derivative of sigmoid is optional:
            known_grads[self.a] \
                = grad_m * self.m_mean * (1. - self.m_mean)
        else:
            known_grads[self.a] = grad_m
            
        params = [self.W[1],self.W[2],self.b[1],self.b[2]]
        
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads,
                       consider_constant=[self.z, self.x],
                       disconnected_inputs='raise')
                       
        updates[self.stoch_grad] = T.abs_(grads[1]).mean()
    
        cost_grads = T.grad(cost=cost, wrt=params,
                       consider_constant=[self.z, self.x],
                       disconnected_inputs='ignore')
                       
        updates[self.kl_grad] = T.abs_(cost_grads[1]).mean()
                       
        for i in range(len(grads)):
            grads[i] += cost_grads[i]
                       
        gradients.update(OrderedDict(izip(params, grads)))
        
        return gradients, updates
        
    def get_kl_divergence(self):
        '''
        Minimize KL-divergence of unit binomial distributions with 
        binomial distribution of probability self.sparsity_target.
        This could also be modified to keep a running average of unit 
        samples
        '''
        e = 1e-6
        cost = - self.sparsity_cost_coeff * ( \
                (self.sparsity_target * T.log(e+self.m_mean.mean(axis=0))) \
                +((1.-self.sparsity_target) * T.log(e+(1.-self.m_mean.mean(axis=0)))) \
             ).sum()
        return cost
        
    def get_weight_decay(self):
        rval = 0
        for i in range(3):
            if self.weight_decay_coeff[i] is not None:
                rval += self.weight_decay_coeff[i]*T.sqr(self.W[i]).sum()
        return rval

    