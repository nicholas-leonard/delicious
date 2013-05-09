__authors__ = "Nicholas Leonard"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Nicholas Leonard"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"
__email__ = "leonardn@iro"

from database import DatabaseHandler

import numpy as np

data_dir = 'dataset/'

def save_terms(db):
    term_keys, term_strings = db.executeSQL("""
    SELECT array_agg(term_key), array_agg(term_string) 
    FROM    (
            SELECT term_key, term_string
            FROM webcluster.term 
            ORDER BY term_doc_count DESC
            ) AS a 
    """, action=db.FETCH_ONE)
    print "we have", len(term_keys), "terms"
    term_indexes = range(len(term_keys))
    term_dict = dict(zip(term_keys, term_indexes))
    term_string = np.asarray(term_strings)

    np.save(data_dir+'corpus_terms', term_strings)
    
    return term_dict
    
def save_dataset(db, term_dict, bucket_range, which_set):
    rows = db.executeSQL("""
    SELECT a.doc_id, c.doc_title, c.doc_url, term_key_array, 
        term_frequency_array, cluster10_key, cluster100_key, 
        cluster1000_key, random()*100000 AS random
    FROM webcluster.doc_term AS a, webcluster.doc_cluster AS b, 
         webcluster.corpus AS c, webcluster.doc_partition AS d
    WHERE a.doc_id = b.doc_id AND a.doc_id = c.doc_id
        AND a.doc_id = d.doc_id AND d.bucket_id BETWEEN %s AND %s
    ORDER BY random
    """, param=bucket_range, action=db.FETCH_ALL) 
    print "we have", len(rows), "examples"
    
    info = []
    inputs = np.zeros((len(rows), len(term_dict)), dtype='float32')
    targets = np.zeros((len(rows), 3), dtype='int32')
    
    for i in xrange(len(rows)):
        doc_id, doc_title, doc_url, term_keys, term_freqs, \
            cluster10_key, cluster100_key, cluster1000_key,_ = rows[i]
        # info
        info.append([doc_id, doc_title, doc_url])
        # inputs
        for (key, freq) in zip(term_keys, term_freqs):
            term_idx = term_dict[key]
            inputs[i,term_idx] = freq
        # targets
        targets[i,:] \
            = np.asarray([cluster10_key,cluster100_key,cluster1000_key])
    
    info = np.asarray(info)
    np.save(data_dir+which_set+'_doc_info', info)
    np.save(data_dir+which_set+'_doc_inputs', inputs)
    np.save(data_dir+which_set+'_doc_targets', targets)
    
def save_all(db):
    term_dict = save_terms(db)
    save_dataset(db, term_dict, (0, 6), 'train')
    save_dataset(db, term_dict, (7, 8), 'valid')
    save_dataset(db, term_dict, (9, 10), 'test')
    
db = DatabaseHandler()
save_all(db)
