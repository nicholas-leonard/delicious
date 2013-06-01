# coding=utf-8

import numpy
import random
import pylab
from database2 import DatabaseHandler
import sys

def get_lcurve_cost(db, config_id):
    rows = db.executeSQL("""
    SELECT 	a.epoch_count, a.channel_value AS train_cost,
        hps3.get_channel(a.config_id::INT4, 'valid_hps_cost'::VARCHAR, a.epoch_count) AS valid_error, 
        hps3.get_channel(a.config_id::INT4, 'test_hps_cost'::VARCHAR, a.epoch_count) AS test_error
    FROM hps3.training_log AS a
    WHERE a.config_id = %s AND a.channel_name = 'train_objective'
    ORDER BY epoch_count ASC
    """,(config_id,),db.FETCH_ALL)
    return numpy.asarray(rows)
    
def get_lcurve_mce(db, config_id):
    rows = db.executeSQL("""
    SELECT 	a.epoch_count,
        1-hps3.get_channel(a.config_id::INT4, 'valid_hps_mca'::VARCHAR, a.epoch_count) AS valid_error, 
        1-hps3.get_channel(a.config_id::INT4, 'test_hps_mca'::VARCHAR, a.epoch_count) AS test_error
    FROM hps3.training_log AS a
    WHERE a.config_id = %s AND a.channel_name = 'train_objective'
    ORDER BY epoch_count ASC    
    """,(config_id,),db.FETCH_ALL)
    return numpy.asarray(rows)

if __name__ == "__main__":
    config_ids = sys.argv[1:-3:2]
    config_names = sys.argv[2:-2:2]
    min_error = float(sys.argv[-2])
    max_error = float(sys.argv[-1])
    db = DatabaseHandler()
        
    errors = [get_lcurve_mce(db, int(config_id)) for config_id in config_ids]
    max_epoch = max(error.shape[0] for error in errors)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    pylab.xlabel('epoch')
    pylab.ylabel('classification error')
    symbols = ['g-2','b-x', 'r-', 'y-+', 'm.', 'k-3', 'c--']
    pylab.axis([0, max_epoch, min_error, max_error])
    for error, name, symbol in zip(errors, config_names, symbols):
        pylab.plot(error[:,0], error[:,1], symbol, label=name, linewidth=1.0)
        #pylab.plot(error[:,0] , error[:,2], label=name+" Test Error")
    pylab.legend()
    fig.savefig('comparison.png', transparent=True)
    pylab.show()

