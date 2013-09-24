#!/usr/bin/env python
#usage: show_weights.py model.pkl
from optparse import OptionParser
import sys
sys.path.append('../')
from stochastic_hps import *
from pylearn2.utils import serial

from pylearn2.utils import serial
from pylearn2.gui import patch_viewer
from pylearn2.config import yaml_parse
from pylearn2.datasets import control
import numpy as np

def get_weights_report(model_path = None, model = None, rescale = 'individual', border = False, norm_sort = False,
        dataset = None):
    """
        Returns a PatchViewer displaying a grid of filter weights

        Parameters:
            model_path: the filepath of the model to make the report on.
            rescale: a string specifying how to rescale the filter images
                        'individual' (default): scale each filter so that it
                            uses as much as possible of the dynamic range
                            of the display under the constraint that 0
                            is gray and no value gets clipped
                        'global' : scale the whole ensemble of weights
                        'none' :   don't rescale
            dataset: a Dataset object to do view conversion for displaying the weights.
                    if not provided one will be loaded from the model's dataset_yaml_src
    """

    if model is None:
        print 'making weights report'
        print 'loading model'
        model = serial.load(model_path)
        print 'loading done'
    else:
        assert model_path is None
    assert model is not None

    if rescale == 'none':
        global_rescale = False
        patch_rescale = False
    elif rescale == 'global':
        global_rescale = True
        patch_rescale = False
    elif rescale == 'individual':
        global_rescale = False
        patch_rescale = True
    else:
        raise ValueError('rescale='+rescale+", must be 'none', 'global', or 'individual'")


    if isinstance(model, dict):
        #assume this was a saved matlab dictionary
        del model['__version__']
        del model['__header__']
        del model['__globals__']
        weights ,= model.values()

        norms = np.sqrt(np.square(weights).sum(axis=1))
        print 'min norm: ',norms.min()
        print 'mean norm: ',norms.mean()
        print 'max norm: ',norms.max()

        return patch_viewer.make_viewer(weights, is_color = weights.shape[1] % 3 == 0)

    weights_view = None
    W = None

    W0,W1,_ = model.get_weights()
    G = model.layers[0].groups
    

    weights_format = ('v', 'g', 'h')

    W1 = W1.T
    W0 = W0.T
    h1 = W1.shape[0]
    h0 = W0.shape[0]
    print W0.shape, W1.shape

    weights_view1 = dataset.get_weights_view(W1)
    weights_view0 = dataset.get_weights_view(W0)

    hr1 = int(np.ceil(np.sqrt(h1)))
    hc1 = hr1
    
    pv1 = patch_viewer.PatchViewer(grid_shape=(hr1,hc1), patch_shape=weights_view1.shape[1:3],
            is_color = weights_view1.shape[-1] == 3)
    
    hr0 = G.shape[0]
    hc0 = int(G.sum(1).max())
    
    pv0 = patch_viewer.PatchViewer(grid_shape=(hr0,hc0), patch_shape=weights_view0.shape[1:3],
            is_color = weights_view0.shape[-1] == 3)
            
    null_patch = np.zeros(weights_view0.shape[1:3])

    if border:
        act = 0
    else:
        act = None

    for i in range(0,h1):
        patch = weights_view1[i,...]
        pv1.add_patch( patch, rescale = patch_rescale, activation = act)
        
    for i in range(0,hr0):
        g = 0
        for j in range(0, G.shape[1]):
            if G[i,j] == 1:
                patch = weights_view0[j,...]
                pv0.add_patch( patch, rescale = patch_rescale, activation = act)
                g += 1
        assert g <= hc0
        for g in range(g,hc0):
            pv0.add_patch( null_patch, rescale = patch_rescale, activation = act)
    return pv0, pv1

def main():
    parser = OptionParser()

    parser.add_option("--rescale",dest='rescale',type='string',default="individual")
    parser.add_option("--border", dest="border", action="store_true",default=False)
    parser.add_option("--dataset", dest='dataset', type='int')

    options, positional = parser.parse_args()

    assert len(positional) == 1
    config_id ,= positional
    
    rescale = options.rescale
    border = options.border
    dataset_id = options.dataset
    
    model_path = '../model_'+str(config_id)+'_optimum.pkl'
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

    pv0, pv1 = get_weights_report(model = model, rescale = rescale, border = border, dataset=dataset)

    pv0.save('weights_'+str(config_id)+'_0.png')
    pv1.save('weights_'+str(config_id)+'_1.png')

if __name__ == "__main__":
    main()
