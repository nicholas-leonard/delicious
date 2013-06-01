# coding=utf-8

import numpy
import random
import pylab
from database import DatabaseHandler
import sys

if __name__ == "__main__":
    config_id = int(sys.argv[1])
    max_error = float(sys.argv[2])
    db = DatabaseHandler()
    rows = db.executeSQL("""
    SELECT 	a.epoch_count, a.channel_value AS train_cost,
        hps3.get_channel(a.config_id::INT4, 'valid_hps_cost'::VARCHAR, a.epoch_count) AS valid_error, 
        hps3.get_channel(a.config_id::INT4, 'test_hps_cost'::VARCHAR, a.epoch_count) AS test_error
    FROM hps3.training_log AS a
    WHERE a.config_id = %s AND a.channel_name = 'train_objective'
    ORDER BY epoch_count ASC
    """,(config_id,),db.FETCH_ALL)
    
    error = numpy.asarray(rows)
    
    pylab.xlabel('epoch')
    pylab.ylabel('error')
    pylab.axis([0, error.shape[0], 0, max_error])
    pylab.plot(error[:,0], error[:,1], 'g', label='Training Error')
    pylab.plot(error[:,0], error[:,2],'r', label='Validation Error')
    pylab.plot(error[:,0] , error[:,3],'b', label="Test Error")
    pylab.legend()
    pylab.show()

