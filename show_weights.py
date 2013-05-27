#!/usr/bin/env python
#usage: show_weights.py model.pkl
import get_weights_report
from optparse import OptionParser
from model4 import *
from pylearn2.utils import serial

def main():
    parser = OptionParser()

    parser.add_option("--rescale",dest='rescale',type='string',default="individual")
    parser.add_option("--out",dest="out",type='string',default=None)
    parser.add_option("--border", dest="border", action="store_true",default=False)
    parser.add_option("--dataset", dest='dataset', type='int')
    parser.add_option("--dim", dest='dim', type='int')

    options, positional = parser.parse_args()

    assert len(positional) == 1
    config_id ,= positional
    
    rescale = options.rescale
    border = options.border
    dataset_id = options.dataset
    
    model_path = 'model_'+str(config_id)+'_optimum.pkl'
    model = serial.load(model_path)
    hps = StochasticHPS('make_submission',-1,None)
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
    
    dataset = hps.train_ddm

    pv = get_weights_report.get_weights_report(model = model, rescale = rescale, border = border, dataset=dataset, dim = options.dim)

    if options.out is None:
        pv.show()
    else:
        pv.save(options.out)

if __name__ == "__main__":
    main()
