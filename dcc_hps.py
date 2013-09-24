__authors__ = "Nicholas Leonard"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Nicholas Leonard"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"

import sys

from hps3 import HPS, HPSData
from stochastic_gater import *
from conditional_gater import *
from group_gater import *
from newsgroups20 import Newsgroups20 
from pylearn2.costs.mlp import WeightDecay, L1WeightDecay

class DCC_HPS(HPS):   
    """
    Distributed Conditional Computation
    Hyper Parameter Search
    """
    def get_layer_stochasticbinaryneuron(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,mean_loss_coeff,hidden_activation,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff,
                 stoch_grad_coeff
        FROM stochastic.layer_StochasticBinaryNeuron
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No StochasticBinaryNeuron for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,mean_loss_coeff,hidden_activation,
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm, weight_decay_coeff, stoch_grad_coeff) = row
        return StochasticBinaryNeuron(dim=dim,hidden_dim=hidden_dim,
                mean_loss_coeff=mean_loss_coeff,istdev=istdev,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,
                stoch_grad_coeff=stoch_grad_coeff)  
                
    def get_layer_gateronly(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,mean_loss_coeff,hidden_activation,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff,
                 stoch_grad_coeff
        FROM stochastic.layer_GaterOnly
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No GaterOnly layer for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,mean_loss_coeff,hidden_activation,
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm, weight_decay_coeff, stoch_grad_coeff) = row
        return GaterOnly(dim=dim,hidden_dim=hidden_dim,
                mean_loss_coeff=mean_loss_coeff,istdev=istdev,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,
                stoch_grad_coeff=stoch_grad_coeff)  
                
    def get_layer_straightthrough(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,hidden_activation,derive_sigmoid,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff,
                 expert_activation
        FROM stochastic.layer_StraightThrough
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No StraightThrough layer for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,hidden_activation,derive_sigmoid,
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm, weight_decay_coeff,expert_activation) = row
        return StraightThrough(dim=dim,hidden_dim=hidden_dim,
                derive_sigmoid=derive_sigmoid,istdev=istdev,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,
                expert_activation=expert_activation)  
           
    def get_layer_smoothtimesstochastic(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,hidden_activation,stochastic_ratio,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 noise_beta, noise_scale, noise_stdev, noise_normality,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff
        FROM stochastic.layer_SmoothTimesStochastic
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No SmoothTimesStochastic layer for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,hidden_activation,stochastic_ratio,
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            noise_beta, noise_scale, noise_stdev, noise_normality,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm,weight_decay_coeff) = row
        return SmoothTimesStochastic(dim=dim,hidden_dim=hidden_dim,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,istdev=istdev,
                noise_beta=noise_beta, noise_scale=noise_scale,
                noise_stdev=noise_stdev,noise_normality=noise_normality,
                stochastic_ratio=stochastic_ratio)  
                
    def get_layer_noisyrectifier(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,hidden_activation,gater_activation,
                 sparsity_cost_coeff,noise_stdev,irange,istdev,
                 sparsity_target,sparsity_decay,expert_activation,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff
        FROM stochastic.layer_NoisyRectifier
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No NoisyRectifier layer for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,hidden_activation,gater_activation,
            sparsity_cost_coeff,noise_stdev,irange,istdev,
            sparsity_target,sparsity_decay,expert_activation,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm,weight_decay_coeff) = row
        return NoisyRectifier(dim=dim,hidden_dim=hidden_dim,
                hidden_activation=hidden_activation,irange=irange,
                init_bias=init_bias,gater_activation=gater_activation,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,istdev=istdev,
                noise_stdev=noise_stdev,sparsity_target=sparsity_target,
                sparsity_decay=sparsity_decay,
                expert_activation=expert_activation) 
                
    def get_layer_baselinesigmoid(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,hidden_activation,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 noise_beta, noise_scale, noise_stdev, noise_normality,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff
        FROM stochastic.layer_BaselineSigmoid
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No BaselineSigmoid layer for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,hidden_activation, 
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            noise_beta, noise_scale, noise_stdev, noise_normality,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm,weight_decay_coeff) = row
        return BaselineSigmoid(dim=dim,hidden_dim=hidden_dim,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,istdev=istdev,
                noise_beta=noise_beta, noise_scale=noise_scale,
                noise_stdev=noise_stdev,noise_normality=noise_normality)  
                
    def get_layer_group1(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   gater_dim,hidden_dim,expert_dim,hidden_activation,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm,weight_decay_coeff,
                 derive_sigmoid,expert_activation
        FROM stochastic.layer_group1
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No group1 layer for layer_id="\
                +str(layer_id))
        (gater_dim,hidden_dim,expert_dim,hidden_activation,
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm,weight_decay_coeff,derive_sigmoid,
            expert_activation) = row
        return Group1(gater_dim=gater_dim,hidden_dim=hidden_dim,
                expert_dim=expert_dim,istdev=istdev,
                expert_activation=expert_activation,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,
                derive_sigmoid=derive_sigmoid)
                
    def get_layer_group2(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,gater_dim,hidden_dim,group_prob,hidden_activation,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm,weight_decay_coeff,
                 derive_sigmoid,expert_activation
        FROM stochastic.layer_group2
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No group1 layer for layer_id="\
                +str(layer_id))
        (dim,gater_dim,hidden_dim,group_prob,hidden_activation,
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm,weight_decay_coeff,derive_sigmoid,
            expert_activation) = row
        return Group2(dim=dim,gater_dim=gater_dim,hidden_dim=hidden_dim,
                group_prob=group_prob,istdev=istdev,
                expert_activation=expert_activation,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,
                derive_sigmoid=derive_sigmoid)
                 
    def get_cost_conditional1(self, cost_id):
        row = self.db.executeSQL("""
        SELECT  cost_type,cost_name,missing_target_value,
                default_dropout_prob,default_dropout_scale
        FROM stochastic.cost_conditional1
        WHERE cost_id = %s
        """, (cost_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No cost for cost_id="+str(cost_id)) 
        (cost_type,cost_name,missing_target_value,
            default_dropout_prob,default_dropout_scale) = row
        mlp_cost = Conditional1Cost(cost_type=cost_type, 
                            missing_target_value=missing_target_value)
        # default monitor based save best channel:
        test_cost = mlp_cost.get_test_cost(self.model,
                                            self.minibatch,
                                            self.target)
        self.add_channel('cost',test_cost)
        
        if self.dropout:
            mlp_cost.setup_dropout(
                default_input_include_prob=(1.-default_dropout_prob),
                default_input_scale=default_dropout_scale,
                input_scales=self.input_scales,
                input_include_probs=self.input_include_probs)
        
        return [mlp_cost]
        
    def get_cost_stochastic1(self, cost_id):
        row = self.db.executeSQL("""
        SELECT  cost_type,cost_name,
                default_dropout_prob,default_dropout_scale
        FROM stochastic.cost_stochastic1
        WHERE cost_id = %s
        """, (cost_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No stochastic1 for cost_id="+str(cost_id)) 
        (cost_type,cost_name,default_dropout_prob, \
            default_dropout_scale) = row
        stochastic_cost = Stochastic1Cost(cost_type=cost_type)
        # default monitor based save best channel:
        test_cost = stochastic_cost.get_test_cost(self.model,
                                                self.minibatch,
                                                self.target)
        self.add_channel('cost',test_cost)
        
        if self.dropout:
            stochastic_cost.setup_dropout(
                default_input_include_prob=(1.-default_dopout_prob),
                default_input_scale=default_dropout_scale,
                input_scales = self.input_scales,
                input_include_probs=self.input_include_probs)
        
        return [stochastic_cost]
        
    def get_layer_stochasticsoftmax(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  n_classes,irange,istdev,sparse_init,W_lr_scale,b_lr_scale, 
                max_row_norm,no_affine,max_col_norm, weight_decay_coeff
        FROM stochastic.layer_stochasticsoftmax
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No stochasticsoftmax layer for layer_id=" \
                +str(layer_id))
        (n_classes,irange,istdev,sparse_init,W_lr_scale,b_lr_scale, 
         max_row_norm,no_affine,max_col_norm,weight_decay_coeff) = row
        return StochasticSoftmax( \
                    n_classes=n_classes,irange=irange,istdev=istdev,
                    sparse_init=sparse_init,W_lr_scale=W_lr_scale,
                    b_lr_scale=b_lr_scale,max_row_norm=max_row_norm,
                    no_affine=no_affine,max_col_norm=max_col_norm,
                    layer_name=layer_name,
                    weight_decay_coeff=weight_decay_coeff)
                    
    def get_layer_tanh(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  dim,irange,istdev,sparse_init,sparse_stdev,include_prob,init_bias,
                W_lr_scale,b_lr_scale,max_col_norm,max_row_norm
        FROM hps3.layer_tanh
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No tanh layer for layer_id=" \
                +str(layer_id))
        (dim,irange,istdev,sparse_init,sparse_stdev,include_prob,init_bias,
            W_lr_scale,b_lr_scale,max_col_norm,max_row_norm) = row
        return SparseTanh(layer_name=layer_name,dim=dim,irange=irange,
                istdev=istdev,sparse_init=sparse_init,
                sparse_stdev=sparse_stdev, include_prob=include_prob,
                init_bias=init_bias,W_lr_scale=W_lr_scale,
                b_lr_scale=b_lr_scale,max_col_norm=max_col_norm,
                max_row_norm=max_row_norm)
                    
    def get_ddm_newsgroups20(self, ddm_id):
        row =  self.db.executeSQL("""
        SELECT which_set,data_path,valid_ratio,sum_to_one,one_hot
        FROM stochastic.ddm_newsgroups20
        WHERE ddm_id = %s
        """, (ddm_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No newsgroups20 ddm for ddm_id="\
                +str(ddm_id))
        (which_set,data_path,valid_ratio,sum_to_one,one_hot) = row
        return Newsgroups20(which_set=which_set,data_path=data_path,
                    valid_ratio=valid_ratio,sum_to_one=sum_to_one,
                    one_hot=one_hot)
                    
                    
if __name__ == '__main__':
    worker_name = str(sys.argv[1])
    task_id = int(sys.argv[2])
    start_config_id = None
    if len(sys.argv) > 3:
        start_config_id = int(sys.argv[3])
    hps = DCC_HPS(task_id=task_id, worker_name=worker_name )
    hps.run(start_config_id)
    if len(sys.argv) < 2:
        print """
        Usage: python hps3.py "worker_name" "task_id" ["config_id"]
        """
