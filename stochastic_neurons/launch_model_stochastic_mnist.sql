/* Stochastic neuron MNIST*/


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,NULL],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{0.01,0.1,1.0}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{NULL,5.0}'::FLOAT4[]) AS mcn) AS f
)RETURNING layer_id;--7753-7758

UPDATE stochastic.layer_stochastic2 SET maWHERE layer_id BETWEEN 7753 AND 7758 AND max_col_norm[1] = 0


INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange) (
	SELECT 'stochasticsoftmax',10,0.05
) RETURNING layer_id;--7759

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8784 AND 8789
UPDATE stochastic.layer_conditional5 SET max_col_norm=ARRAY[5.0,5.0,NULL] WHERE layer_id BETWEEN 7747 AND 7752
UPDATE hps3.config_mlp_sgd SET description='stoch2m' WHERE task_id = 37;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',37,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7759]::INT8[],'{8}'::INT8[],
		32,lr,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond5 easy'
	FROM (SELECT generate_series(7753,7758,1) AS layer1) AS a,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS lr) AS b
) RETURNING config_id;--8784-8789

/* different lr for output layer */


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,NULL],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{10,100}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{NULL,5.0}'::FLOAT4[]) AS mcn) AS f
)RETURNING layer_id;--7760-7763

UPDATE stochastic.layer_stochastic2 SET max_col_norm = ARRAY[NULL,NULL,NULL]::FLOAT4[] WHERE layer_id BETWEEN 7753 AND 7758 AND max_col_norm[1] = 0


INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange,w_lr_scale,b_lr_scale) (
	SELECT 'stochasticsoftmax',10,0.05,lrs,lrs
	FROM (SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lrs) AS a
) RETURNING layer_id;--7764-7765

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8784 AND 8789
UPDATE stochastic.layer_conditional5 SET max_col_norm=ARRAY[5.0,5.0,NULL] WHERE layer_id BETWEEN 7747 AND 7752
UPDATE hps3.config_mlp_sgd SET cost_array = '{8}'::INT8[] WHERE task_id = 37;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',38,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2n lrs'
	FROM (SELECT generate_series(7760,7763,1) AS layer1) AS a,
		(SELECT generate_series(7764,7765,1) AS layer2) AS b
) RETURNING config_id;--8796-8803


/*  smaller lr */


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1,5}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{5.0}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7766-7769

UPDATE stochastic.layer_stochastic2 SET max_col_norm = ARRAY[NULL,NULL,NULL]::FLOAT4[] WHERE layer_id BETWEEN 7753 AND 7758 AND max_col_norm[1] = 0


INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange,w_lr_scale,b_lr_scale) (
	SELECT 'stochasticsoftmax',10,0.05,lrs,lrs
	FROM (SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lrs) AS a
) RETURNING layer_id;--7764-7765

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8784 AND 8789
UPDATE stochastic.layer_conditional5 SET max_col_norm=ARRAY[5.0,5.0,NULL] WHERE layer_id BETWEEN 7747 AND 7752
UPDATE hps3.config_mlp_sgd SET start_time = now() WHERE config_id = 8807;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',39,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2o lrs'
	FROM (SELECT generate_series(7766,7769,1) AS layer1) AS a,
		(SELECT generate_series(7764,7765,1) AS layer2) AS b
) RETURNING config_id;--8804-8811


/*  smaller lr */


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{NULL}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7770-7771

UPDATE stochastic.layer_stochastic2 SET max_col_norm = ARRAY[NULL,NULL,NULL]::FLOAT4[] WHERE layer_id BETWEEN 7753 AND 7758 AND max_col_norm[1] = 0


INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange,w_lr_scale,b_lr_scale) (
	SELECT 'stochasticsoftmax',10,0.05,lrs,lrs
	FROM (SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lrs) AS a
) RETURNING layer_id;--7764-7765

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8784 AND 8789
UPDATE stochastic.layer_conditional5 SET max_col_norm=ARRAY[5.0,5.0,NULL] WHERE layer_id BETWEEN 7747 AND 7752
UPDATE hps3.config_mlp_sgd SET start_time = now() WHERE config_id = 8807;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',40,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2o lrs'
	FROM (SELECT generate_series(7770,7771,1) AS layer1) AS a
) RETURNING config_id;--8812-8813



/*  smaller lr */


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7772-7773

UPDATE stochastic.layer_stochastic2 SET max_col_norm = ARRAY[NULL,NULL,NULL]::FLOAT4[] WHERE layer_id BETWEEN 7753 AND 7758 AND max_col_norm[1] = 0


INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange,w_lr_scale,b_lr_scale) (
	SELECT 'stochasticsoftmax',10,0.05,lrs,lrs
	FROM (SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lrs) AS a
) RETURNING layer_id;--7764-7765

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8784 AND 8789
UPDATE stochastic.layer_conditional5 SET max_col_norm=ARRAY[5.0,5.0,NULL] WHERE layer_id BETWEEN 7747 AND 7752
UPDATE hps3.config_mlp_sgd SET start_time = now() WHERE config_id = 8807;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',40,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2p mcn'
	FROM (SELECT generate_series(7772,7773,1) AS layer1) AS a
) RETURNING config_id;--8814-8815




/*  exponential decay */


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',40,18,4,'{1}'::INT8[],
		'{1,8}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2q expodecay'
	FROM (SELECT generate_series(7770,7773,1) AS layer1) AS a
) RETURNING config_id;--8816-8819


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',40,18,4,'{1}'::INT8[],
		'{1,10}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2q expodecay++'
	FROM (SELECT generate_series(7770,7773,1) AS layer1) AS a
) RETURNING config_id;--8832-8835



/*  more dept */


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7783


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / (idim + dim)), sqrt(6. / (idim +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{200,2000}'::FLOAT4[]) AS idim) AS h
)RETURNING layer_id;--7784-7785

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',42,18,4,'{1}'::INT8[],
		'{1,8}'::INT8[],'{5,4}'::INT8[],ARRAY[7783,layer2, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2r dept++'
	FROM (SELECT generate_series(7784,7785,1) AS layer2) AS a
) RETURNING config_id;--8838-8839

UPDATE hps3.config_mlp_sgd SET task_id = 42 WHERE config_id BETWEEN 8838 AND 8839



/* Multiply stochastic gradient by constant < 1?  */


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1,0.1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7789-7792


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / (idim + dim)), sqrt(6. / (idim +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1,0.1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{200}'::FLOAT4[]) AS idim) AS h
)RETURNING layer_id;--7793-7796

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',42,18,4,'{1}'::INT8[],
		'{1,8}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1,layer2, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2r dept++'
	FROM (SELECT generate_series(7789,7792,1) AS layer1) AS a,
		(SELECT generate_series(7793,7796,1) AS layer2) AS b
) RETURNING config_id;--8842-8857

UPDATE hps3.config_mlp_sgd SET task_id = 42 WHERE config_id BETWEEN 8838 AND 8839


/* stochastic gradient coefficient */


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.7}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.01,0.001}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7798-7801

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2s sgc'
	FROM (SELECT generate_series(7798,7801,1) AS layer1) AS a
) RETURNING config_id;--8864-8867


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.1,0.1],ARRAY[lrs,0.1,0.1],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.7}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7802-7805


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2t sgc lrs'
	FROM (SELECT generate_series(7802,7805,1) AS layer1) AS a
) RETURNING config_id;--8868-8871


/* No momentum */

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2t no momentum'
	FROM (SELECT generate_series(7804,7805,1) AS layer1) AS a
) RETURNING config_id;--8872-8873

/* Higner lr for linear part */


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.1,0.1],ARRAY[lrs,0.1,0.1],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.7}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0,10.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7806-7807


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2t no momentum'
	FROM (SELECT generate_series(8874,8875,1) AS layer1) AS a
) RETURNING config_id;--8874-8875



/* More or less capacity */


INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.1,0.1],ARRAY[lrs,0.1,0.1],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{1000,4000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.7}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7808-7809



INSERT INTO stochastic.layer_stochastic2 (
layer_class,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.1,0.1],ARRAY[lrs,0.1,0.1],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.05,0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.7}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7810--7811

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2u capacity'
	FROM (SELECT generate_series(7808,7811,1) AS layer1) AS a
) RETURNING config_id;--8876-8879

/* no tanh */

INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2',NULL,dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.1,0.1],ARRAY[lrs,0.1,0.1],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.7}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7812

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[7812, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2v notanh'
) RETURNING config_id;--8880


/* no sgc */

INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7814

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[7814, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2v no sgc'
) RETURNING config_id;--8881


/* expodecay++ */

INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.1,0.1],ARRAY[lrs,0.1,0.1],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7815

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{8}'::INT8[],'{5,4}'::INT8[],ARRAY[7815, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2v expodecay++'
) RETURNING config_id;--8883


/* warped capacity */

INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{200}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{2,4,8}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7816-7818

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2w warped capacity'
	FROM (SELECT generate_series(7816,7818,1) AS layer1) AS a
) RETURNING config_id;--8884-8886



/* sparsity target */

INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,st,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.01,0.05,0.2}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--7819-7821

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2x sparsity target'
	FROM (SELECT generate_series(7819,7821,1) AS layer1) AS a
) RETURNING config_id;--8887-8889

UPDATE hps3.config_mlp_sgd SET start_time = NULL WHERE config_id = 8882
SELECT * FROM hps3.config_mlp_sgd WHERE config_id = 8882



/* warped capacity */

INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,0.1,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{200}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{4}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7822

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[7822, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2w warped capacity / more time'
) RETURNING config_id;--8890


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{8}'::INT8[],'{5,6}'::INT8[],ARRAY[7822, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2w warped capacity / more time / expodecay++'
) RETURNING config_id;--8890



/* sparsity target */

INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,st,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{1000,4000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.01,0.05,0.2}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--7823-7828

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',43,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2x sparsity target / capacity'
	FROM (SELECT generate_series(7823,7828,1) AS layer1) AS a
) RETURNING config_id;--8892-8897



/* smaller model */

INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,st,scc,ARRAY[mcn,mcn,mcn*10],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{200}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.05,0.1,0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1,1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.05,0.1,0.2}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--7985-8002

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',54,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch2y smaller model'
	FROM (SELECT generate_series(7985,8002,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--9089-9106