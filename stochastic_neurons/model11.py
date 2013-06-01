__authors__ = "Nicholas Leonard"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Nicholas Leonard"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"

import sys

from stochastic_hps import *
         
        
if __name__=='__main__':
    worker_name = str(sys.argv[1])
    task_id = int(sys.argv[2])
    start_config_id = None
    if len(sys.argv) > 3:
        start_config_id = int(sys.argv[3])
    base_channel_names = \
         ['train_objective',
          'train_stochastic30_max_unit_sparsity_prop',
          'train_stochastic30_mean_output_sparsity',
          'train_stochastic30_mean_sparsity_prop', 
          'train_stochastic30_min_unit_sparsity_prop',
          'train_stochastic30_mean_unit_sparsity_meta_prop',
          'train_stochastic30_mean_unit_sparsity_meta_prop2',
          'train_stochastic30_mean_sparsity_prop0.2',
          'train_stochastic30_mean_sparsity_prop0.3',
          'train_stochastic30_mean_sparsity_prop0.4',
          'train_stochastic30_output_stdev',
          'train_stochastic30_output_meta_stdev',
          'train_stochastic30_mean_loss_nume_mean',
          'train_stochastic30_mean_loss_deno_mean',
          'train_stochastic30_stoch_grad',
          'train_stochastic30_kl_grad']
          
    hps = StochasticHPS(task_id=task_id, worker_name=worker_name,
                        base_channel_names=base_channel_names)
                        
    hps.run(start_config_id)
    if len(sys.argv) < 2:
        print """
        Usage: python test1.py "worker_name" "task_id" ["config_id"]
        """
