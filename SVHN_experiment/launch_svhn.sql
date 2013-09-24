
INSERT INTO hps3.ddm_svhn(ddm_class,which_set)
VALUES 	('svhn','splitted_train'),('svhn','valid'),('svhn','test')
RETURNING ddm_id;--144-146

INSERT INTO hps3.preprocess(preprocess_class) VALUES ('lcn') RETURNING preprocess_id;--8

INSERT INTO hps3.dataset(preprocess_array,train_ddm_id,valid_ddm_id,test_ddm_id)
VALUES (ARRAY[2,8]::INT4[],144,145,146) RETURNING dataset_id;--60

INSERT INTO hps3.dataset(train_ddm_id,valid_ddm_id,test_ddm_id)
VALUES (144,145,146) RETURNING dataset_id;--61

SELECT * FROM hps3.config_mlp_sgd WHERE dataset_id = 60

UPDATE hps3.config_mlp_sgd 
SET dataset_id = 61, train_iteration_mode = 'batchwise_shuffled_equential' 
WHERE config_id = 11241
WHERE task_id = 88 AND start_time IS NULL
