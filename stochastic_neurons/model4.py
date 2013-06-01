__authors__ = "Nicholas Leonard"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Nicholas Leonard"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"

import sys

from hps3 import HPS, HPSData
from stochastic_gater import *
from conditional_gater import *
from newsgroups20 import Newsgroups20 
from pylearn2.costs.mlp import WeightDecay, L1WeightDecay


"""
TODO:
    finish hps loaders
    test baseline mlp with 20 newsgroups
"""

class StochasticHPS(HPS):
    def get_layer_stochastic1(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,mean_loss_coeff,hidden_activation,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff
        FROM stochastic.layer_stochastic1
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No stochastic1 layer for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,mean_loss_coeff,hidden_activation,
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm, weight_decay_coeff) = row
        return Stochastic1(dim=dim,hidden_dim=hidden_dim,
                mean_loss_coeff=mean_loss_coeff,istdev=istdev,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff)   
                
    def get_layer_stochastic2(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,mean_loss_coeff,hidden_activation,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff
        FROM stochastic.layer_stochastic2
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No stochastic1 layer for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,mean_loss_coeff,hidden_activation,
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm, weight_decay_coeff) = row
        return Stochastic2(dim=dim,hidden_dim=hidden_dim,
                mean_loss_coeff=mean_loss_coeff,istdev=istdev,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff)  
                
    def get_layer_conditional1(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,hidden_activation,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff
        FROM stochastic.layer_conditional1
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No conditional1 layer for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,hidden_activation,
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm, weight_decay_coeff) = row
        return Conditional1(dim=dim,hidden_dim=hidden_dim,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,istdev=istdev)  
                
    def get_layer_conditional2(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   dim,hidden_dim,hidden_activation,
                 sparsity_target,sparsity_cost_coeff,irange,istdev,
                 variance_beta, variance_cost_coeff,
                 sparse_init,sparse_stdev,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, weight_decay_coeff
        FROM stochastic.layer_conditional2
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No conditional2 layer for layer_id="\
                +str(layer_id))
        (dim,hidden_dim,hidden_activation, 
            sparsity_target,sparsity_cost_coeff,irange,istdev,
            variance_beta,variance_cost_coeff,
            sparse_init,sparse_stdev,init_bias,W_lr_scale,b_lr_scale,
            max_col_norm,weight_decay_coeff) = row
        return Conditional2(dim=dim,hidden_dim=hidden_dim,
                hidden_activation=hidden_activation,irange=irange,
                sparsity_target=sparsity_target,init_bias=init_bias,
                sparsity_cost_coeff=sparsity_cost_coeff,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_col_norm=max_col_norm,layer_name=layer_name,
                weight_decay_coeff=weight_decay_coeff,istdev=istdev,
                variance_beta=variance_beta,
                variance_cost_coeff=variance_cost_coeff)  
                 
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
        
        costs = [mlp_cost]
        if self.weight_decay:
            coeffs = []
            for layer in self.mlp.layers:
                coeffs.append(self.weight_decays[layer.layer_name])
            wd_cost = WeightDecay(coeffs)
            costs.append(wd_cost)
        if self.l1_weight_decay:
            coeffs = []
            for layer in self.mlp.layers:
                coeffs.append(self.l1_weight_decays[layer.layer_name])
            lwd_cost = L1WeightDecay(coeffs)
            costs.append(lwd_cost)
        return costs
                                
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
        
        costs = [stochastic_cost]
        if self.weight_decay:
            coeffs = []
            for layer in self.mlp.layers:
                coeffs.append(self.weight_decays[layer.layer_name])
            wd_cost = WeightDecay(coeffs)
            costs.append(wd_cost)
        if self.l1_weight_decay:
            coeffs = []
            for layer in self.mlp.layers:
                coeffs.append(self.l1_weight_decays[layer.layer_name])
            lwd_cost = L1WeightDecay(coeffs)
            costs.append(lwd_cost)
        return costs
        
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
    
        
if __name__=='__main__':
    worker_name = str(sys.argv[1])
    task_id = int(sys.argv[2])
    start_config_id = None
    if len(sys.argv) > 3:
        start_config_id = int(sys.argv[3])
    base_channel_names = \
         ['train_objective', 
          'train_conditional20_max_unit_sparsity_prop',
          'train_conditional20_mean_output_sparsity',
          'train_conditional20_mean_sparsity_prop', 
          'train_conditional20_min_unit_sparsity_prop',
          'train_conditional20_mean_unit_sparsity_meta_prop',
          'train_conditional20_mean_unit_sparsity_meta_prop2',
          'train_conditional20_mean_sparsity_prop0.2',
          'train_conditional20_mean_sparsity_prop0.3',
          'train_conditional20_mean_sparsity_prop0.4',
          'train_conditional20_output_stdev',
          'train_conditional20_output_meta_stdev']
          
    hps = StochasticHPS(task_id=task_id, worker_name=worker_name,
                        base_channel_names=base_channel_names)
    hps.run(start_config_id)
    if len(sys.argv) < 2:
        print """
        Usage: python test1.py "worker_name" "task_id" ["config_id"]
        """
