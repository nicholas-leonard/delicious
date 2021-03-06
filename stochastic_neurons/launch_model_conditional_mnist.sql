﻿

INSERT INTO stochastic.cost_conditional1 (cost_class,cost_type) VALUES ('conditional1', 'default') RETURNING cost_id;--9

INSERT INTO hps3.ddm_mnist(ddm_class,which_set,start,stop)
VALUES 	('mnist','train',0,50000),('mnist','train',50000,60000),('mnist','test',NULL,NULL)
RETURNING ddm_id;--18-20

INSERT INTO hps3.dataset(preprocess_array,train_ddm_id,valid_ddm_id,test_ddm_id)
VALUES (ARRAY[1]::INT4[],18,19,20) RETURNING dataset_id;--17

INSERT INTO hps3.channel (channel_class) VALUES ('mca');--1

INSERT INTO hps3.layer_softmax (layer_class,n_classes,irange,W_lr_scale,b_lr_scale) (
	SELECT 'softmax',10,0.005,lr_scale, lr_scale
	FROM (SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale) AS a
) RETURNING layer_id;--6448-6451


INSERT INTO stochastic.layer_conditional2 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,variance_beta,variance_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT 'conditional2', dim, (dim*0.2)::INT4,0.1,sparsity_cost_coeff,variance_beta,variance_cost_coeff, c, d, d
FROM	(
	SELECT  dim, random(),variance_beta, variance_cost_coeff,sparsity_cost_coeff,
		ARRAY[0.005,0.005,0.005]::FLOAT4[] as c,ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{500,1000,2000}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{1,10,100}'::FLOAT4[]) AS variance_cost_coeff) AS b,
		(SELECT unnest('{0.6,1.1,2.1}'::FLOAT4[]) AS variance_beta) AS d,
		(SELECT unnest('{1,10,100}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 100 ) AS a
)RETURNING layer_id;--6954-7053

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',26,17,4,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{9}'::INT8[],
		32,0.5,NULL,random()*1000000 AS random_seed, 'cond2a'
	FROM (SELECT generate_series(6954,7053,1) AS layer1) AS a,
	     (SELECT generate_series(7054,7055,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 50
) RETURNING config_id;--7759-7808

UPDATE hps3.config_mlp_sgd SET task_id = 26 WHERE config_id BETWEEN 7759 AND 7808

BEGIN;
DELETE FROM hps3.config_mlp_sgd 
USING stochastic.layer_conditional1 AS a
WHERE layer_array[1] = a.layer_id AND start_time IS NULL
AND (a.W_lr_scale[1] = 0.001::FLOAT4 OR a.W_lr_scale[2] = 0.001::FLOAT4 OR a.W_lr_scale[3] = 0.001::FLOAT4) AND task_id = 24;
COMMIT;

UPDATE hps3.config_mlp_sgd SET description = 'stoch2b no NaNs? new channels' WHERE task_id = 23 AND start_time IS NULL

SELECT COUNT(*) FROM hps3.config_mlp_sgd WHERE task_id = 26 AND end_time IS NOT NULL;

/* Beta 0.6-- ; lr_scale around 0.01++ ; less expo decay; 
sparsity cost 100 >> variance cost coeff 1; dim ++; */



INSERT INTO stochastic.layer_conditional2 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,variance_beta,variance_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT 'conditional2', dim, (dim*0.2)::INT4,0.1,sparsity_cost_coeff,variance_beta,variance_cost_coeff, c, d, d
FROM	(
	SELECT  dim, random(),variance_beta, variance_cost_coeff,sparsity_cost_coeff,
		ARRAY[0.005,0.005,0.005]::FLOAT4[] as c,ARRAY[lr_scale1,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{2000,4000,8000}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,1,5}'::FLOAT4[]) AS variance_cost_coeff) AS b,
		(SELECT unnest('{0.11,0.21,0.41,0.51}'::FLOAT4[]) AS variance_beta) AS d,
		(SELECT unnest('{10,100}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 150 ) AS a
)RETURNING layer_id;--7056-7205


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',27,17,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{9}'::INT8[],
		32,0.5,NULL,random()*1000000 AS random_seed, 'cond2b'
	FROM (SELECT generate_series(7056,7205,1) AS layer1) AS a,
	     (SELECT generate_series(7054,7055,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 150
) RETURNING config_id;--7959-8108

DELETE FROM hps3.config_mlp_sgd WHERE task_id = 27

/* higher beta... */

INSERT INTO stochastic.layer_conditional2 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,variance_beta,variance_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT 'conditional2', dim, (dim*0.2)::INT4,0.1,sparsity_cost_coeff,variance_beta,variance_cost_coeff, c, d, d
FROM	(
	SELECT  dim, random(),variance_beta, variance_cost_coeff,sparsity_cost_coeff,
		ARRAY[0.005,0.005,0.005]::FLOAT4[] as c,ARRAY[lr_scale1,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{4000}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0,1,10}'::FLOAT4[]) AS variance_cost_coeff) AS b,
		(SELECT unnest('{10.1,20.1,40.1,80.1}'::FLOAT4[]) AS variance_beta) AS d,
		(SELECT unnest('{10,100,1000}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 150 ) AS a
)RETURNING layer_id;--7206-7349


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',27,17,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{9}'::INT8[],
		32,0.5,NULL,random()*1000000 AS random_seed, 'cond2c'
	FROM (SELECT generate_series(7206,7349,1) AS layer1) AS a,
	     (SELECT generate_series(7054,7055,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 150
) RETURNING config_id;--8109-8258

DELETE FROM hps3.config_mlp_sgd WHERE config_id BETWEEN 7959 AND 8108 AND start_time IS NULL;

/* aarons stochastic conditional */


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT 'conditional3', dim, (dim*0.2)::INT4,0.1,sparsity_cost_coeff,variance_beta,variance_cost_coeff, c, d, d
FROM	(
	SELECT  dim, random(),sparsity_cost_coeff,
		ARRAY[0.005,0.005,0.005]::FLOAT4[] as c,ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{4000,8000,16000}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{10,100,1000}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 150 ) AS a
)RETURNING layer_id;--7206-7349


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',27,17,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{9}'::INT8[],
		32,0.5,NULL,random()*1000000 AS random_seed, 'cond2c'
	FROM (SELECT generate_series(7206,7349,1) AS layer1) AS a,
	     (SELECT generate_series(7054,7055,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 150
) RETURNING config_id;--8109-8258



/* Mnist smaller sparsitycost and learning rates */

INSERT INTO stochastic.layer_conditional2 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,variance_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT 'conditional2', dim, (dim*0.2)::INT4,0.1,sparsity_cost_coeff, 0, c, d, d
FROM	(
	SELECT  dim, random(), sparsity_cost_coeff,
		ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (0.2*dim))), 4*sqrt(6. / ((28*28) + dim))]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{8000}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,1,10}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 150 ) AS a
)RETURNING layer_id;--7350-7430


INSERT INTO hps3.dataset(train_ddm_id,valid_ddm_id,test_ddm_id) VALUES (18,19,20) RETURNING dataset_id;--18

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',28,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond2c'
	FROM (SELECT generate_series(7350,7430,1) AS layer1) AS a,
	     (SELECT generate_series(7054,7055,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 162
) RETURNING config_id;--8259-8420

/* Mnist smaller sparsitycost */

INSERT INTO stochastic.layer_conditional2 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,variance_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT 'conditional2', dim, (dim*0.2)::INT4,0.1,sparsity_cost_coeff, 0, c, d, d
FROM	(
	SELECT  dim, random(), sparsity_cost_coeff,
		ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (0.2*dim))), 4*sqrt(6. / ((28*28) + dim))]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{8000}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.01,0.001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 150 ) AS a
)RETURNING layer_id;--7431-7484

DELETE FROM hps3.config_mlp_sgd WHERE task_id = 28 AND start_time IS NULL;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',28,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond2d'
	FROM (SELECT generate_series(7431,7484,1) AS layer1) AS a,
	     (SELECT generate_series(7054,7055,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 30
) RETURNING config_id;--8421-8450


/* Mnist l1 norm constraint relu */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,sparse_init,irange,W_lr_scale,b_lr_scale) (
SELECT 'conditional4', dim, (dim*0.2)::INT4,'rectifiedlinear',sparsity_cost_coeff, e, c, d, d
FROM	(
	SELECT  dim, random(), sparsity_cost_coeff,
		ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (0.2*dim))), NULL]::FLOAT4[] as c,
		ARRAY[NULL,NULL,15]::INT4[] AS e,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{8000}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.001,0.0001,0.00001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 50 ) AS a
)RETURNING layer_id;--7535-7558

DELETE FROM hps3.config_mlp_sgd WHERE task_id = 29;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',29,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond3a'
	FROM (SELECT generate_series(7535,7558,1) AS layer1) AS a,
	     (SELECT generate_series(7054,7055,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 30
) RETURNING config_id;--8481-8510


/* Mnist higher and equal lrs */

INSERT INTO stochastic.layer_conditional2 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,variance_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT 'conditional2', dim, (dim*0.2)::INT4,0.1,0.001, 0, c, d, d
FROM	(
	SELECT  dim, random(),
		ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (0.2*dim))), 4*sqrt(6. / ((28*28) + dim))]::FLOAT4[] as c,
		ARRAY[lr_scale,lr_scale,lr_scale]::FLOAT[] as d
	FROM (SELECT unnest('{8000}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lr_scale) AS g
	ORDER BY random
	LIMIT 150 ) AS a
)RETURNING layer_id;--7559-7560


DELETE FROM hps3.config_mlp_sgd WHERE task_id = 30 AND start_time IS NULL;


INSERT INTO hps3.layer_softmax (layer_class,n_classes,irange,W_lr_scale,b_lr_scale) (
	SELECT 'softmax',10,0.005,lr_scale, lr_scale
	FROM (SELECT unnest('{1.0,0.1}'::FLOAT4[]) AS lr_scale) AS a
) RETURNING layer_id;--7561-7562

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',30,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[7559, 7561]::INT8[],'{9}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond2d'
	UNION ALL
	SELECT 'mlp','sgd',30,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[7560, 7562]::INT8[],'{9}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond2d'
) RETURNING config_id;--8511-8512

SELECT * FROM stochastic.layer_conditional2 WHERE layer_id = 7559


/* Mnist l1 norm constraint relu */

INSERT INTO hps3.layer_softmax (layer_class,n_classes,irange) (
	SELECT 'softmax',10,0.005
) RETURNING layer_id;--7572

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,sparse_init,irange) (
SELECT 'conditional4', dim, (dim*0.2)::INT4,'rectifiedlinear',sparsity_cost_coeff, e, c
FROM	(
	SELECT  dim, random(), sparsity_cost_coeff,
		ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (0.2*dim))), NULL]::FLOAT4[] as c,
		ARRAY[NULL,NULL,15]::INT4[] AS e
	FROM (SELECT unnest('{8000}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.00001,0.000001,0.0000001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e
	ORDER BY random
	LIMIT 50 ) AS a
)RETURNING layer_id;--7569-7571

DELETE FROM hps3.config_mlp_sgd WHERE task_id = 31;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',31,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,lr,NULL,random()*1000000 AS random_seed, 'cond3a'
	FROM (SELECT generate_series(7569,7571,1) AS layer1) AS a,
	     (SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr) AS b
	ORDER BY random_seed LIMIT 30
) RETURNING config_id;--8513-8518


/* Mnist less gater capacity */

INSERT INTO stochastic.layer_conditional2 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,variance_cost_coeff,irange) (
SELECT 'conditional2', 8000, hidden_dim,0.1,0.001, 0, c
FROM	(
	SELECT  hidden_dim, random(),
		ARRAY[sqrt(6. / ((28*28) + 8000)), sqrt(6. / ((28*28) + (0.2*8000))), 4*sqrt(6. / ((28*28) + 8000))]::FLOAT4[] as c
	FROM (SELECT unnest('{25,100,400}'::INT4[]) AS hidden_dim) AS a
	ORDER BY random
	LIMIT 150 ) AS a
)RETURNING layer_id;--7574-7576

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',32,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,lr,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond2e'
	FROM (SELECT generate_series(7574,7576,1) AS layer1) AS a,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr) AS b
) RETURNING config_id;--8519-8524

SELECT * FROM stochastic.layer_conditional2 WHERE layer_id = 7559



/* Mnist beta noise */

INSERT INTO stochastic.layer_conditional5 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,noise_beta,irange) (
SELECT 'conditional5', 8000, 1600,0.1,0.001,80.1, 
	ARRAY[sqrt(6. / ((28*28) + 8000)), sqrt(6. / ((28*28) + (0.2*8000))), 4*sqrt(6. / ((28*28) + 8000))]::FLOAT4[]
)RETURNING layer_id;--7577

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',33,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[7577, 7572]::INT8[],'{9}'::INT8[],
		32,lr,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond2e'
	FROM (SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr) AS b
) RETURNING config_id;--8525-8526

UPDATE hps3.config_mlp_sgd SET start_time = NULL WHERE config_id = 8526

UPDATE hps3.config_mlp_sgd SET layer_array[2] = 7572 WHERE config_id BETWEEN 8527 AND 8535


INSERT INTO stochastic.layer_conditional5 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,noise_beta,irange) (
SELECT 'conditional5', 8000, 1600,0.1,0.001,noise_beta, 
	ARRAY[sqrt(6. / ((28*28) + 8000)), sqrt(6. / ((28*28) + (0.2*8000))), 4*sqrt(6. / ((28*28) + 8000))]::FLOAT4[]
FROM (SELECT unnest('{1.1,10.1,20.1,40.1}'::FLOAT4[]) AS noise_beta) AS a
)RETURNING layer_id;--7578-7581

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',33,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond2f'
	FROM (SELECT generate_series(7578,7581,1) AS layer1) AS a
	UNION ALL
	SELECT 'mlp','sgd',33,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond2f'
	FROM (SELECT generate_series(7577,7581,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8527-8535

/* Gaussian noise + Beta noise */

INSERT INTO stochastic.layer_conditional5 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,noise_beta,noise_stdev,noise_normality,irange) (
SELECT 'conditional5', dim, dim*hdp,0.1,0.001,40.1,7.0,noise_normality,
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{4000,8000,16000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.05,0.1,0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0,0.5,1.0}'::FLOAT4[]) AS noise_normality) AS c
)RETURNING layer_id;--7615-7641

UPDATE stochastic.layer_conditional5 SET noise_stdev = 0.5 WHERE layer_id BETWEEN 7615 AND 7641

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',34,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond2h'
	FROM (SELECT generate_series(7615,7641,1) AS layer1) AS a
	UNION ALL
	SELECT 'mlp','sgd',34,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond2h'
	FROM (SELECT generate_series(7615,7641,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8623-8676

SELECT * FROM hps3.config_mlp_sgd WHERE task_id = 33

UPDATE hps3.config_mlp_sgd SET start_time=NULL WHERE config_id BETWEEN 8623 AND 8676

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8623 AND 8676



/* Aaron's model */

INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,stochastic_ratio,40.1,0.0,noise_normality,
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{4000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{0.1,0.5,0.9}'::FLOAT4[]) AS stochastic_ratio) AS d
)RETURNING layer_id;--7642-7644

UPDATE stochastic.layer_conditional5 SET noise_stdev = 0.5 WHERE layer_id BETWEEN 7615 AND 7641

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3a no-noise'
	FROM (SELECT generate_series(7642,7644,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8677-8679


/* noise */

INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,40.1,noise_stdev,noise_normality,
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{4000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{5.0,10.0,20.0}'::FLOAT4[]) AS noise_stdev) AS d
UNION ALL
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,noise_beta,0,noise_normality,
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{4000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{5.1,20.1,40.1}'::FLOAT4[]) AS noise_beta) AS d
)RETURNING layer_id;--7645-7650


UPDATE stochastic.layer_conditional5 SET noise_stdev = 0.5 WHERE layer_id BETWEEN 7615 AND 7641

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3b noise'
	FROM (SELECT generate_series(7645,7650,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8680-8685

SELECT * FROM hps3.config_mlp_sgd WHERE task_id = 35 AND start_time IS NULL

UPDATE hps3.config_mlp_sgd SET task_id = 35 WHERE config_id BETWEEN 8677 AND 8679

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8623 AND 8676

INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,40.1,noise_stdev,noise_normality,
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{4000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{0.01,0.1,0.25}'::FLOAT4[]) AS noise_stdev) AS d
)RETURNING layer_id;--7654-7656


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3b noise'
	FROM (SELECT generate_series(7654,7656,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8686-8688

/* normal noise and hiddens++*/


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,40.1,noise_stdev,noise_normality,
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{8000,16000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_stdev) AS d
)RETURNING layer_id;--7657-7658


UPDATE stochastic.layer_conditional5 SET noise_stdev = 0.5 WHERE layer_id BETWEEN 7615 AND 7641

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3c normal size++'
	FROM (SELECT generate_series(7657,7658,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8680-8685



/* normal noise and hiddens++ and max_col_norms */


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,40.1,noise_stdev,noise_normality,ARRAY[2.5,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{8000,16000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_stdev) AS d
)RETURNING layer_id;--7659-7660


UPDATE stochastic.layer_conditional5 SET noise_stdev = 0.5 WHERE layer_id BETWEEN 7615 AND 7641

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3c normal size++'
	FROM (SELECT generate_series(7659,7660,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8680-8685



/* normal noise and hiddens++ and max_col_norms */


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,40.1,noise_stdev,noise_normality,ARRAY[2.5,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{8000,16000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{0.75,0.5}'::FLOAT4[]) AS noise_stdev) AS d
)RETURNING layer_id;--7665-7668


UPDATE stochastic.layer_conditional5 SET noise_stdev = 0.5 WHERE layer_id BETWEEN 7615 AND 7641

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3d normal size++ noise_stdev--'
	FROM (SELECT generate_series(7665,7668,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8680-8685



/* normal noise and hiddens++ and max_col_norms */


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,noise_beta,0,noise_normality,ARRAY[2.5,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{8000,16000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{5.1,40.1,80.1}'::FLOAT4[]) AS noise_beta) AS d
)RETURNING layer_id;--7669-7674


UPDATE stochastic.layer_conditional5 SET noise_stdev = 0.5 WHERE layer_id BETWEEN 7615 AND 7641

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3e beta size++'
	FROM (SELECT generate_series(7669,7674,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8680-8685



/* before going to bed, launch lots of jobs */


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,noise_beta,0,noise_normality,ARRAY[cn,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000,4000,8000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{5.1,40.1,80.1}'::FLOAT4[]) AS noise_beta) AS d,
	(SELECT unnest('{5.0,NULL}'::FLOAT4[]) AS cn) AS e
UNION ALL
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,40.1,noise_stdev,noise_normality,ARRAY[cn,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000,4000,8000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{0.1,0.25,0.5}'::FLOAT4[]) AS noise_stdev) AS d,
	(SELECT unnest('{5.0,NULL}'::FLOAT4[]) AS cn) AS e
)RETURNING layer_id;--7675-7746


UPDATE stochastic.layer_conditional5 SET noise_stdev = 0.5 WHERE layer_id BETWEEN 7615 AND 7641

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3f*'
	FROM (SELECT generate_series(7675,7746,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8706-8777

/* Conditional baseline: same computational capacity*/

INSERT INTO stochastic.layer_conditional5 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,noise_beta,noise_stdev,noise_normality,noise_scale,max_col_norm,irange) (
SELECT 'conditional5', dim, dim*hdp,0,0,NULL,noise_stdev,1.0,1.0,ARRAY[5.0,5.0,NULL]
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{200}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{1,2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0,1.0,7.0}'::FLOAT4[]) AS noise_stdev) AS c
)RETURNING layer_id;--7747-7752

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8778 AND 8783
UPDATE stochastic.layer_conditional5 SET max_col_norm=ARRAY[5.0,5.0,NULL] WHERE layer_id BETWEEN 7747 AND 7752

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',36,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,lr,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond5 easy'
	FROM (SELECT generate_series(7747,7752,1) AS layer1) AS a,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS lr) AS b
) RETURNING config_id;--8778-8783


/* Conditional baseline: same computational capacity*/

INSERT INTO stochastic.layer_conditional5 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,noise_beta,noise_stdev,noise_normality,noise_scale,max_col_norm,irange) (
SELECT 'conditional5', dim, dim*hdp,0,0,NULL,noise_stdev,1.0,1.0,ARRAY[5.0,5.0,NULL]
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{200}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{1,2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0,1.0,7.0}'::FLOAT4[]) AS noise_stdev) AS c
)RETURNING layer_id;--7747-7752

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8778 AND 8783
UPDATE stochastic.layer_conditional5 SET max_col_norm=ARRAY[5.0,5.0,NULL] WHERE layer_id BETWEEN 7747 AND 7752

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',36,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,lr,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond5 easy no-momentum'
	FROM (SELECT generate_series(7747,7752,1) AS layer1) AS a,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS lr) AS b
) RETURNING config_id;--8790-8795



/* less capacity, momentum? */


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,noise_beta,0,noise_normality,ARRAY[cn,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{500,1000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{40.1,80.1,160.1}'::FLOAT4[]) AS noise_beta) AS d,
	(SELECT unnest('{NULL}'::FLOAT4[]) AS cn) AS e
)RETURNING layer_id;--7774-7779

UPDATE hps3.config_mlp_sgd SET start_time = now() WHERE task_id = 35 AND start_time IS NULL;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3g cap--'
	FROM (SELECT generate_series(7774,7779,1) AS layer1) AS a
	UNION ALL
	SELECT 'mlp','sgd',35,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3g cap-- nm'
	FROM (SELECT generate_series(7774,7779,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--8820-8831



/* dept++ */


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,noise_beta,0,noise_normality,ARRAY[cn,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{40.1}'::FLOAT4[]) AS noise_beta) AS d,
	(SELECT unnest('{5}'::FLOAT4[]) AS cn) AS e
)RETURNING layer_id;--7780


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,noise_beta,0,noise_normality,ARRAY[cn,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / (idim + dim)), sqrt(6. / (idim +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{200,2000}'::FLOAT4[]) AS idim) AS f,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{40.1}'::FLOAT4[]) AS noise_beta) AS d,
	(SELECT unnest('{5}'::FLOAT4[]) AS cn) AS e
)RETURNING layer_id;--7781-7782

UPDATE hps3.config_mlp_sgd SET start_time = now() WHERE task_id = 35 AND start_time IS NULL;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',41,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[7780,layer2, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3h dept++'
	FROM (SELECT generate_series(7781,7782,1) AS layer2) AS a
) RETURNING config_id;--8836-8837

UPDATE hps3.config_mlp_sgd SET init_momentum = 0.5 WHERE config_id BETWEEN 8836 AND 8837;




/* dept++ no noise */


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,40.1,0,1.0,ARRAY[cn,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2}'::FLOAT4[]) AS cn) AS e
)RETURNING layer_id;--7788


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,40.1,0,1.0,ARRAY[cn,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / (idim + dim)), sqrt(6. / (idim +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{200}'::FLOAT4[]) AS idim) AS f,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{5}'::FLOAT4[]) AS cn) AS e
)RETURNING layer_id;--7787

UPDATE stochastic.layer_conditional3 SET noise_beta = 40.1 WHERE layer_id BETWEEN 7788 AND 7788

UPDATE hps3.config_mlp_sgd SET start_time = now() WHERE task_id = 35 AND start_time IS NULL;

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',41,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[7788,7787,7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3h dept++ no noise'
) RETURNING config_id;--8841



INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',41,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[7788,7787,7787,7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3h dept+++ no noise'
) RETURNING config_id;--8858


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',41,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[7788,7787,7787,7787,7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3h dept+++ no noise'
) RETURNING config_id;--8859


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',41,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[7788,7787,7787,7787,7787,7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3h dept+++ no noise'
) RETURNING config_id;--8860



INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',41,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[7788,7787,7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3h dept++ gradient halted'
) RETURNING config_id;--8861


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,40.1,0,1.0,ARRAY[cn,NULL,NULL]::FLOAT4[],
	ARRAY[sqrt(6. / (idim + dim)), sqrt(6. / (idim +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{4000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{200}'::FLOAT4[]) AS idim) AS f,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{5}'::FLOAT4[]) AS cn) AS e
)RETURNING layer_id;--7797

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',41,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[7788,7797,7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3h dept++ gradient halted'
) RETURNING config_id;--8862


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',41,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[7788,7788,7797,7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond3h dept++ gradient halted'
) RETURNING config_id;--8863


/* l1 norm constraint stochastic relu */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,noise_stdev,sparse_init,W_lr_scale,b_lr_scale,max_col_norm,irange) (
SELECT  'conditional4', dim, (dim*hdp)::INT4,'rectifiedlinear',sparsity_cost_coeff,ns,
	ARRAY[NULL,NULL,15],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (hdp*dim))), NULL]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS c,
	(SELECT unnest('{0,0.5}'::FLOAT4[]) AS ns ) AS d, 
	(SELECT unnest('{0.0001,0.00001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7923-7926

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',48,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5,random()*1000000 AS random_seed, 'cond4a'
	FROM (SELECT generate_series(7923,7926,1) AS layer1) AS a
) RETURNING config_id;--9033-9036

UPDATE hps3.config_mlp_sgd SET init_momentum = 0.5 WHERE task_id = 48


/* l1 norm constraint-- stochastic relu */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,noise_stdev,sparse_init,W_lr_scale,b_lr_scale,max_col_norm,irange) (
SELECT  'conditional4', dim, (dim*hdp)::INT4,'rectifiedlinear',sparsity_cost_coeff,ns,
	ARRAY[NULL,NULL,15],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (hdp*dim))), NULL]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS c,
	(SELECT unnest('{0,0.5}'::FLOAT4[]) AS ns ) AS d, 
	(SELECT unnest('{0.000001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7923-7926

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',48,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5,random()*1000000 AS random_seed, 'cond4a'
	FROM (SELECT generate_series(7927,7928,1) AS layer1) AS a
) RETURNING config_id;--9037-9038


/* no momentum stochastic relu */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,noise_stdev,sparse_init,W_lr_scale,b_lr_scale,max_col_norm,irange) (
SELECT  'conditional4', dim, (dim*hdp)::INT4,'rectifiedlinear',sparsity_cost_coeff,ns,
	ARRAY[NULL,NULL,15],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (hdp*dim))), NULL]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS c,
	(SELECT unnest('{0,0.5}'::FLOAT4[]) AS ns ) AS d, 
	(SELECT unnest('{0.00001,0.000001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7929-7932

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',48,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond4b no momentum'
	FROM (SELECT generate_series(7929,7932,1) AS layer1) AS a
) RETURNING config_id;--9039-9042


/* noise++ */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,noise_stdev,sparse_init,W_lr_scale,b_lr_scale,max_col_norm,irange) (
SELECT  'conditional4', dim, (dim*hdp)::INT4,'rectifiedlinear',sparsity_cost_coeff,ns,
	ARRAY[NULL,NULL,15],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (hdp*dim))), NULL]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS c,
	(SELECT unnest('{1.0,5.0}'::FLOAT4[]) AS ns ) AS d, 
	(SELECT unnest('{0.00001,0.000001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7933-7936

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',48,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond4c noise++'
	FROM (SELECT generate_series(7933,7936,1) AS layer1) AS a
) RETURNING config_id;--9043-9046



/* sparsity target */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,noise_stdev,
sparsity_target,sparsity_decay,sparse_init,W_lr_scale,b_lr_scale,max_col_norm,irange) (
SELECT  'conditional4', dim, (dim*hdp)::INT4,'rectifiedlinear',sparsity_cost_coeff,ns,st,sd,
	ARRAY[NULL,NULL,15],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (hdp*dim))), NULL]
FROM (SELECT unnest('{4000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS c,
	(SELECT unnest('{0.5,1.0,2.0}'::FLOAT4[]) AS ns ) AS d, 
	(SELECT unnest('{0.000001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS f,
	(SELECT unnest('{0.95,0.9}'::FLOAT4[]) AS sd) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7937-7942

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',49,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond4d sparsity_decay'
	FROM (SELECT generate_series(7937,7942,1) AS layer1) AS a
) RETURNING config_id;--9047-9052



/* sparsity decay */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,noise_stdev,
sparsity_target,sparsity_decay,sparse_init,W_lr_scale,b_lr_scale,max_col_norm,irange) (
SELECT  'conditional4', dim, (dim*hdp)::INT4,'rectifiedlinear',sparsity_cost_coeff,ns,st,sd,
	ARRAY[NULL,NULL,15],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (hdp*dim))), NULL]
FROM (SELECT unnest('{1000,2000,4000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS c,
	(SELECT unnest('{0.5,1.0,2.0}'::FLOAT4[]) AS ns ) AS d, 
	(SELECT unnest('{0.00001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS f,
	(SELECT unnest('{0.1,0.05}'::FLOAT4[]) AS sd) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--7967-7984

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',53,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond4f sparsity_decay, capacity'
	FROM (SELECT generate_series(7967,7984,1) AS layer1) AS a
) RETURNING config_id;--9071-9088

UPDATE hps3.config_mlp_sgd SET task_id = 50 WHERE config_id BETWEEN 9053 AND 9058

/* Conditional baseline: same computational capacity / smaller model */

INSERT INTO stochastic.layer_conditional5 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,noise_beta,noise_stdev,noise_normality,noise_scale,max_col_norm,irange) (
SELECT 'conditional5', dim, dim*hdp,0,0,NULL,noise_stdev,1.0,1.0,ARRAY[mcn,mcn,mcn*10],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{20}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.5,1,2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0,1.0,7.0}'::FLOAT4[]) AS noise_stdev) AS c,
	(SELECT unnest('{2.1,1}'::FLOAT4[]) AS mcn) AS d
)RETURNING layer_id;--8027-8020

DELETE FROM hps3.training_log WHERE config_id BETWEEN 8778 AND 8783
UPDATE stochastic.layer_conditional5 SET max_col_norm=ARRAY[5.0,5.0,NULL] WHERE layer_id BETWEEN 7747 AND 7752

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',55,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,lr,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'cond5 smaller model'
	FROM (SELECT generate_series(8003,8020,1) AS layer1) AS a,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS lr) AS b
) RETURNING config_id;--9107-9124


/* relu baselines */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,noise_stdev,
sparsity_target,sparsity_decay,sparse_init,W_lr_scale,b_lr_scale,max_col_norm,irange) (
SELECT  'conditional4', dim, (dim*hdp)::INT4,'rectifiedlinear',sparsity_cost_coeff,ns,st,sd,
	ARRAY[NULL,NULL,15],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (hdp*dim))), NULL]
FROM (SELECT unnest('{1000,2000,4000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS c,
	(SELECT unnest('{0}'::FLOAT4[]) AS ns ) AS d, 
	(SELECT unnest('{0.00001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS f,
	(SELECT unnest('{0.1,0.05}'::FLOAT4[]) AS sd) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--8045-8050


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',53,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond4f baseline'
	FROM (SELECT generate_series(8046,8050,1) AS layer1) AS a
) RETURNING config_id;--9125-9130

/* more capacity */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,noise_stdev,
sparsity_target,sparsity_decay,sparse_init,W_lr_scale,b_lr_scale,max_col_norm,irange) (
SELECT  'conditional4', dim, (dim*hdp)::INT4,'rectifiedlinear',sparsity_cost_coeff,ns,st,sd,
	ARRAY[NULL,NULL,15],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (hdp*dim))), NULL]
FROM (SELECT unnest('{8000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.025,0.05,0.1,0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS c,
	(SELECT unnest('{0.0,0.5,1.0,2.0}'::FLOAT4[]) AS ns ) AS d, 
	(SELECT unnest('{0.00001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
	(SELECT unnest('{0.05,0.1}'::FLOAT4[]) AS st) AS f,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS sd) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--8083-8114

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',53,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond4 more capacity'
	FROM (SELECT generate_series(8083,8114,1) AS layer1) AS a
) RETURNING config_id;--9131-9162

SELECT * FROM hps3.config_mlp_sgd WHERE task_id = 53

UPDATE hps3.config_mlp_sgd SET task_id = 50 WHERE config_id BETWEEN 9053 AND 9058