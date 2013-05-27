
INSERT INTO stochastic.ddm_newsgroups20(ddm_class,which_set)
VALUES 	('newsgroups20','train'),('newsgroups20','valid'),('newsgroups20','test')
RETURNING ddm_id;--12-14

INSERT INTO hps3.dataset(train_ddm_id,valid_ddm_id,test_ddm_id)
VALUES (12,13,14) RETURNING dataset_id;--15

INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange)
VALUES ('stochasticsoftmax',20,0.05) RETURNING layer_id;--589

INSERT INTO stochastic.layer_stochastic1 (layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,irange) (
	SELECT 'stochastic1',dim,(dim*hidden_proportion)::INT4,mean_loss_coeff,
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[]
	FROM (SELECT unnest('{40,80,160,320,640}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.2,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.1,0.5,1.0}'::FLOAT4[]) AS sparsity_cost_coeff) AS e
)RETURNING layer_id;--590-949


INSERT INTO stochastic.cost_stochastic1 (cost_class,cost_type) VALUES ('stochastic1', 'default') RETURNING cost_id;--8

INSERT INTO hps3.channel (channel_class) VALUES ('mca');--1

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed) (
	SELECT 'mlp','sgd',18,15,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 589]::INT8[],'{8}'::INT8[],
		32,lr,0.5,random()*1000000 AS random_seed
	FROM (SELECT unnest('{.1,.01,.001,.0001}'::FLOAT4[]) AS lr) AS a,
		(SELECT generate_series(590,949,1) AS layer1) AS b
	ORDER BY random_seed DESC LIMIT 20
) RETURNING config_id;--3815-3834

--UPDATE hps3.config_mlp_sgd SET start_time=NULL WHERE config_id BETWEEN 3815 AND 3834;
--BEGIN; DELETE FROM hps3.config_mlp_sgd WHERE start_time IS NULL AND config_id BETWEEN 3815 AND 3834; COMMIT;
--DELETE FROM hps3.training_log WHERE config_id BETWEEN 42 AND 95

/* Baseline MLP */


INSERT INTO hps3.layer_tanh (layer_class, dim, irange) (
	SELECT 'tanh',dim,0.05 FROM (SELECT unnest('{40,80,160,320}'::INT4[]) AS dim) AS a
)RETURNING layer_id;--950-953

INSERT INTO hps3.layer_softmax(layer_class, n_classes, irange) VALUES (20, 0.05) RETURNING layer_id;--954


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,random_seed) (
	SELECT 'mlp','sgd',19,15,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 954]::INT8[],'{4}'::INT8[],
		32,lr,random()*1000000 AS random_seed
	FROM (SELECT unnest('{.1,.01,.001,.0001}'::FLOAT4[]) AS lr) AS a,
		(SELECT generate_series(950,953,1) AS layer1) AS b
	ORDER BY random_seed 
) RETURNING config_id;--3835-3850

UPDATE hps3.config_mlp_sgd SET init_momentum = 0.5 WHERE task_id = 19

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,random_seed) (
	SELECT 'mlp','sgd',19,15,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 954]::INT8[],'{4}'::INT8[],
		32,lr,random()*1000000 AS random_seed
	FROM (SELECT unnest('{.1,.01,.001,.0001}'::FLOAT4[]) AS lr) AS a,
		(SELECT generate_series(950,953,1) AS layer1) AS b
	ORDER BY random_seed 
) RETURNING config_id;--3851-3866

INSERT INTO stochastic.ddm_newsgroups20(ddm_class,which_set,sum_to_one)
VALUES 	('newsgroups20','train',false),('newsgroups20','valid',false),('newsgroups20','test',false)
RETURNING ddm_id;--15-17

INSERT INTO hps3.dataset(train_ddm_id,valid_ddm_id,test_ddm_id)
VALUES (15,16,17) RETURNING dataset_id;--16

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,random_seed) (
	SELECT 'mlp','sgd',19,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 954]::INT8[],'{4}'::INT8[],
		32,lr,random()*1000000 AS random_seed
	FROM (SELECT unnest('{.1,.01}'::FLOAT4[]) AS lr) AS a,
		(SELECT generate_series(950,953,1) AS layer1) AS b
	ORDER BY random_seed LIMIT 2
) RETURNING config_id;--3867-8

/* Stochastic */

INSERT INTO stochastic.layer_stochastic1 (layer_class,dim,hidden_dim,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,irange) (
	SELECT 'stochastic1',dim,(dim*hidden_proportion)::INT4,mean_loss_coeff,
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[]
	FROM (SELECT unnest('{40,80,160,320,640}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.2,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS sparsity_cost_coeff) AS e
)RETURNING layer_id;--955-1314


INSERT INTO stochastic.cost_stochastic1 (cost_class,cost_type) VALUES ('stochastic1', 'default') RETURNING cost_id;--8

INSERT INTO hps3.channel (channel_class) VALUES ('mca');--1

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',20,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 589]::INT8[],'{8}'::INT8[],
		32,lr,NULL,random()*1000000 AS random_seed, 'stoch1 no momentum'
	FROM (SELECT unnest('{.1,.01,.001}'::FLOAT4[]) AS lr) AS a,
		(SELECT generate_series(955,1314,1) AS layer1) AS b
	UNION ALL
	SELECT 'mlp','sgd',20,16,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 589]::INT8[],'{8}'::INT8[],
		32,lr,0.5,random()*1000000 AS random_seed, 'stoch1 momentum'
	FROM (SELECT unnest('{.1,.01,.001}'::FLOAT4[]) AS lr) AS a,
		(SELECT generate_series(955,1314,1) AS layer1) AS b
	ORDER BY random_seed DESC LIMIT 20
) RETURNING config_id;--3869-3888


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',20,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 589]::INT8[],'{8}'::INT8[],
		32,lr,NULL,random()*1000000 AS random_seed, 'stoch1 no momentum'
	FROM (SELECT unnest('{.001,.0001}'::FLOAT4[]) AS lr) AS a,
		(SELECT generate_series(955,1314,1) AS layer1) AS b
	UNION ALL
	SELECT 'mlp','sgd',20,16,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 589]::INT8[],'{8}'::INT8[],
		32,lr,0.5,random()*1000000 AS random_seed, 'stoch1 momentum'
	FROM (SELECT unnest('{.001,.0001}'::FLOAT4[]) AS lr) AS a,
		(SELECT generate_series(955,1314,1) AS layer1) AS b
	ORDER BY random_seed DESC LIMIT 20
) RETURNING config_id;--3889-3908

--BEGIN; UPDATE hps3.config_mlp_sgd SET description='stoch1 momentum' WHERE config_id BETWEEN 3869 AND 3908 AND description='stoch2 momentum'; COMMIT;
--BEGIN; DELETE FROM hps3.config_mlp_sgd WHERE start_time IS NULL AND task_id=20; COMMIT;
--DELETE FROM hps3.training_log WHERE config_id BETWEEN 42 AND 95

INSERT INTO stochastic.layer_stochastic1 (layer_class,dim,hidden_dim,mean_loss_coeff,weight_decay,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b, mean_loss_coeff,weight_decay,sparsity_target,sparsity_cost_coeff, c, d, d
FROM	(
	SELECT 'stochastic1' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,weight_decay,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{40,80,160,320,640,1280}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.01,0.1,1.0,10}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.00001,0.0001,0.001,0.01}'::FLOAT4[]) AS weight_decay) AS f,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 200 ) AS a
)RETURNING layer_id;--3048-3247

INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange,W_lr_scale,b_lr_scale) (
	SELECT 'stochasticsoftmax',20,0.05,lr_scale, lr_scale
	FROM (SELECT unnest('{1.0,0.1,0.01,0.001,0.0001}'::FLOAT4[]) AS lr_scale) AS a
) RETURNING layer_id;--3043-3047

/* Weight decay to keep from Nan */

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',20,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'stoch1b no momentum'
	FROM (SELECT generate_series(3048,3247,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	UNION ALL
	SELECT 'mlp','sgd',20,16,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,0.5,random()*1000000 AS random_seed, 'stoch1b momentum'
	FROM (SELECT generate_series(3048,3247,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	ORDER BY random_seed DESC LIMIT 800
) RETURNING config_id;--3909-4708

SELECT * FROM hps3.config_mlp_sgd WHERE config_id BETWEEN 3909 AND 4708 AND start_time = NULL
UPDATE hps3.config_mlp_sgd SET start_time=NULL WHERE config_id=4218

/* Max col norm on stochasticW0*/

INSERT INTO stochastic.layer_stochastic1 (layer_class,dim,hidden_dim,mean_loss_coeff,weight_decay,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale,max_col_norm) (
SELECT a, dim, b, mean_loss_coeff,weight_decay,sparsity_target,sparsity_cost_coeff, c, d, d, e
FROM	(
	SELECT 'stochastic1' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,weight_decay,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d, ARRAY[max_col_norm,NULL,NULL] AS e
	FROM (SELECT unnest('{40,80,160,320,640,1280}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.01,0.1,1.0,10}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.00001,0.0001,0.001,0.01}'::FLOAT4[]) AS weight_decay) AS f,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i,
		(SELECT unnest('{0.1,10,100}'::FLOAT4[]) AS max_col_norm) AS j
	ORDER BY random
	LIMIT 400 ) AS a
)RETURNING layer_id;--3248-3647


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',20,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'stoch1c no momentum'
	FROM (SELECT generate_series(3248,3647,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	UNION ALL
	SELECT 'mlp','sgd',20,16,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,0.5,random()*1000000 AS random_seed, 'stoch1c momentum'
	FROM (SELECT generate_series(3248,3647,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	ORDER BY random_seed DESC LIMIT 1600
) RETURNING config_id;--4709-6308

/* News integrated layer-wise costs */


INSERT INTO stochastic.layer_stochastic1 (layer_class,dim,hidden_dim,mean_loss_coeff,weight_decay_coeff,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b, mean_loss_coeff,f,sparsity_target,sparsity_cost_coeff, c, d, d
FROM	(
	SELECT 'stochastic1' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,
		ARRAY[wdc0,wdc1,wdc2]::FLOAT4[] AS f,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{40,80,160,320,640,1280}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.01,0.1,1.0,10}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.00001,0.0001,0.001,0.01}'::FLOAT4[]) AS wdc0) AS f,
		(SELECT unnest('{0.00001,0.0001,0.001,0.01}'::FLOAT4[]) AS wdc1) AS k,
		(SELECT unnest('{0.00001,0.0001,0.001,0.01}'::FLOAT4[]) AS wdc2) AS j,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 400 ) AS a
)RETURNING layer_id;--3648-4047


INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange,W_lr_scale,b_lr_scale) (
	SELECT 'stochasticsoftmax',20,0.05,lr_scale, lr_scale
	FROM (SELECT unnest('{1.0,0.1,0.01,0.001,0.0001}'::FLOAT4[]) AS lr_scale) AS a
) RETURNING layer_id;--3043-3047


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',21,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'stoch1d no momentum'
	FROM (SELECT generate_series(3648,4047,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	UNION ALL
	SELECT 'mlp','sgd',21,16,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,0.5,random()*1000000 AS random_seed, 'stoch1d momentum'
	FROM (SELECT generate_series(3648,4047,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	ORDER BY random_seed DESC LIMIT 800
) RETURNING config_id;--6309-7108

--DELETE FROM hps3.config_mlp_sgd WHERE task_id = 21 AND start_time is NULL;

/* Fixed KL-divergence */


INSERT INTO stochastic.layer_stochastic1 (layer_class,dim,hidden_dim,mean_loss_coeff,weight_decay_coeff,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b, mean_loss_coeff,f,sparsity_target,sparsity_cost_coeff, c, d, d
FROM	(
	SELECT 'stochastic1' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,
		ARRAY[wdc0,wdc1,wdc2]::FLOAT4[] AS f,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{40,80,160,320}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.01,0.1,1.0,10}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.00001,0.0001,0.001,0.01}'::FLOAT4[]) AS wdc0) AS f,
		(SELECT unnest('{0.00001,0.0001,0.001,0.01}'::FLOAT4[]) AS wdc1) AS k,
		(SELECT unnest('{0.00001,0.0001,0.001,0.01}'::FLOAT4[]) AS wdc2) AS j,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 400 ) AS a
)RETURNING layer_id;--4048-4447


INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange,W_lr_scale,b_lr_scale) (
	SELECT 'stochasticsoftmax',20,0.05,lr_scale, lr_scale
	FROM (SELECT unnest('{1.0,0.1,0.01,0.001,0.0001}'::FLOAT4[]) AS lr_scale) AS a
) RETURNING layer_id;--3043-3047


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',22,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'stoch1e no momentum'
	FROM (SELECT generate_series(4048,4447,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	UNION ALL
	SELECT 'mlp','sgd',22,16,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,0.5,random()*1000000 AS random_seed, 'stoch1e momentum'
	FROM (SELECT generate_series(4048,4447,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	ORDER BY random_seed DESC LIMIT 100
) RETURNING config_id;--7109-7208

/* emphasize sparsity */

INSERT INTO stochastic.layer_stochastic1 (layer_class,dim,hidden_dim,mean_loss_coeff,weight_decay_coeff,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b, mean_loss_coeff,f,sparsity_target,sparsity_cost_coeff, c, d, d
FROM	(
	SELECT 'stochastic1' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,
		ARRAY[wdc0,wdc1,wdc2]::FLOAT4[] AS f,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{40,80,160,320}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{1.0,10,100,1000}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc0) AS f,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc1) AS k,
		(SELECT unnest('{0.00001,0.0001,0.001,0.01,0.1}'::FLOAT4[]) AS wdc2) AS j,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 400 ) AS a
)RETURNING layer_id;--4848-5247

DELETE FROM hps3.config_mlp_sgd WHERE config_id BETWEEN 7109 AND 7208 AND start_time IS NULL;


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',22,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'stoch1f no momentum'
	FROM (SELECT generate_series(4848,5247,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	UNION ALL
	SELECT 'mlp','sgd',22,16,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,0.5,random()*1000000 AS random_seed, 'stoch1f momentum'
	FROM (SELECT generate_series(4848,5247,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	ORDER BY random_seed DESC LIMIT 100
) RETURNING config_id;--7209-7308

UPDATE hps3.config_mlp_sgd SET description = 'stoch1g T.grad' WHERE task_id = 22 AND start_time IS NULL

/* De-emphasize sparsity (really fixed kl this time) */

INSERT INTO stochastic.layer_stochastic1 (layer_class,dim,hidden_dim,mean_loss_coeff,weight_decay_coeff,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b, mean_loss_coeff,f,sparsity_target,sparsity_cost_coeff, c, d, d
FROM	(
	SELECT 'stochastic1' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,
		ARRAY[wdc0,wdc1,wdc2]::FLOAT4[] AS f,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{40,80,160}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.001,0.01,0.1,1.0}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc0) AS f,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc1) AS k,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc2) AS j,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 400 ) AS a
)RETURNING layer_id;--5248-5647

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',22,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'stoch1h no momentum'
	FROM (SELECT generate_series(5248,5647,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 100
) RETURNING config_id;--7309-7408

/* col norms */

INSERT INTO stochastic.layer_stochastic1 (layer_class,dim,hidden_dim,mean_loss_coeff,max_col_norm,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b, mean_loss_coeff,f,sparsity_target,sparsity_cost_coeff, c, d, d
FROM	(
	SELECT 'stochastic1' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,
		ARRAY[wdc0,wdc1,wdc2]::FLOAT4[] AS f,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{40,80,160}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.001,0.01,0.1,1.0}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{1,10,100}'::FLOAT4[]) AS wdc0) AS f,
		(SELECT unnest('{1,10,100}'::FLOAT4[]) AS wdc1) AS k,
		(SELECT unnest('{1,10,100}'::FLOAT4[]) AS wdc2) AS j,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 400 ) AS a
)RETURNING layer_id;--5648-6047

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',22,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'stoch1i no momentum'
	FROM (SELECT generate_series(5648,6047,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 100
) RETURNING config_id;--7409-7508

SELECT COUNT(*) FROM hps3.config_mlp_sgd WHERE start_time IS NULL AND task_id = 23

/* Stochastic2 */


INSERT INTO stochastic.layer_stochastic2 (layer_class,dim,hidden_dim,mean_loss_coeff,weight_decay_coeff,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b, mean_loss_coeff,f,sparsity_target,sparsity_cost_coeff, c, d, d
FROM	(
	SELECT 'stochastic2' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,
		ARRAY[wdc0,wdc1,wdc2]::FLOAT4[] AS f,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{40,80,160}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.01,0.1,1.0,10}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc0) AS f,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc1) AS k,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc2) AS j,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 400 ) AS a
)RETURNING layer_id;--6048-6447

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',23,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'stoch2a no momentum'
	FROM (SELECT generate_series(6048,6447,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 100
) RETURNING config_id;--7509-7608

/* Stochastic2 More hiddens */


INSERT INTO stochastic.layer_stochastic2 (layer_class,dim,hidden_dim,mean_loss_coeff,weight_decay_coeff,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b, mean_loss_coeff,f,sparsity_target,sparsity_cost_coeff, c, d, d
FROM	(
	SELECT 'stochastic2' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,
		ARRAY[wdc0,wdc1,wdc2]::FLOAT4[] AS f,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{40,80,160}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1,0.4,0.8}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{0.01,0.1,1.0,10}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc0) AS f,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc1) AS k,
		(SELECT unnest('{0.001,0.01,0.1}'::FLOAT4[]) AS wdc2) AS j,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 400 ) AS a
)RETURNING layer_id;--6048-6447

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',23,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{8}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'stoch2a no momentum'
	FROM (SELECT generate_series(6048,6447,1) AS layer1) AS a,
	     (SELECT generate_series(3043,3047,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 100
) RETURNING config_id;--7509-7608

UPDATE hps3.config_mlp_sgd SET description = 'stoch2b no NaNs? new channels' WHERE task_id = 23 AND start_time IS NULL



/* Stochastic2 More hiddens */


INSERT INTO stochastic.layer_stochastic2 (layer_class,dim,hidden_dim,mean_loss_coeff,
					 sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b, mean_loss_coeff,0.1,sparsity_cost_coeff, ARRAY[0.05,0.05,0.05]::FLOAT4[], d, d
FROM	(
	SELECT 'stochastic2' AS a,dim,(dim*hidden_proportion)::INT4 AS b,mean_loss_coeff,random(),
		sparsity_cost_coeff,ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{320,640,1280}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.7,0.8,0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
		(SELECT unnest('{5,10,20}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{1.0,0.1,0.01}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{1.0,0.1,0.01}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 100 ) AS a
)RETURNING layer_id;--6852-6951


INSERT INTO stochastic.layer_stochasticsoftmax (layer_class,n_classes,irange,W_lr_scale,b_lr_scale) (
	SELECT 'stochasticsoftmax',20,0.05,lr_scale, lr_scale
	FROM (SELECT unnest('{0.01,0.001}'::FLOAT4[]) AS lr_scale) AS a
) RETURNING layer_id;--6952-6953


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',25,16,3,'{1}'::INT8[],
		ext_array,'{5,4}'::INT8[],layer_array,'{8}'::INT8[],
		32,0.1,init_momentum,random_seed, 'stoch2c capacity++'
	FROM	(
		SELECT ARRAY[layer1, layer2]::INT8[] AS layer_array, random()*1000000 AS random_seed
		FROM (SELECT generate_series(6852,6951,1) AS layer1) AS a,
		     (SELECT generate_series(6952,6953,1) AS layer2) AS b
		ORDER BY random_seed LIMIT 25 
		) AS a,
		(
			SELECT '{1,6}'::INT8[] AS ext_array, 0.2 AS init_momentum
			UNION ALL
			SELECT '{6}'::INT8[] AS ext_array, NULL AS init_momentum
		) AS b
) RETURNING config_id;--7758;7709

SELECT MAX(config_id), MIN(config_id) FROM hps3.config_mlp_sgd WHERE task_id = 25