

INSERT INTO stochastic.cost_conditional1 (cost_class,cost_type) VALUES ('conditional1', 'default') RETURNING cost_id;--9

INSERT INTO hps3.channel (channel_class) VALUES ('mca');--1

INSERT INTO hps3.layer_softmax (layer_class,n_classes,irange,W_lr_scale,b_lr_scale) (
	SELECT 'softmax',20,0.05,lr_scale, lr_scale
	FROM (SELECT unnest('{1.0,0.1,0.01,0.001}'::FLOAT4[]) AS lr_scale) AS a
) RETURNING layer_id;--6448-6451


UPDATE hps3.config_mlp_sgd SET description = 'stoch1g T.grad' WHERE task_id = 22 AND start_time IS NULL


INSERT INTO stochastic.layer_conditional1 (
	layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,irange,W_lr_scale,b_lr_scale) (
SELECT a, dim, b,sparsity_target,sparsity_cost_coeff, c, d, d
FROM	(
	SELECT 'conditional1' AS a,dim,(dim*hidden_proportion)::INT4 AS b,random(),
		sparsity_target,sparsity_cost_coeff,ARRAY[0.05,0.05,0.05]::FLOAT4[] as c,
		ARRAY[lr_scale0,lr_scale1,lr_scale2]::FLOAT[] as d
	FROM (SELECT unnest('{160,320,640}'::INT4[]) AS dim) AS a,
		(SELECT unnest('{0.1,0.2,0.4}'::FLOAT4[]) AS hidden_proportion) AS b,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS sparsity_target) AS d,
		(SELECT unnest('{1.0,10,100}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
		(SELECT unnest('{1.0,0.1,0.01}'::FLOAT4[]) AS lr_scale0) AS g,
		(SELECT unnest('{1.0,0.1,0.01}'::FLOAT4[]) AS lr_scale1) AS h,
		(SELECT unnest('{1.0,0.1,0.01}'::FLOAT4[]) AS lr_scale2) AS i
	ORDER BY random
	LIMIT 400 ) AS a
)RETURNING layer_id;--6452-6851

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',24,16,3,'{1}'::INT8[],
		'{6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'cond1a no momentum'
	FROM (SELECT generate_series(6452,6851,1) AS layer1) AS a,
	     (SELECT generate_series(6448,6451,1) AS layer2) AS b
	UNION ALL
	SELECT 'mlp','sgd',24,16,3,'{1}'::INT8[],
		'{1,6}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, layer2]::INT8[],'{9}'::INT8[],
		32,0.1,0.5,random()*1000000 AS random_seed, 'cond1a momentum'
	FROM (SELECT generate_series(6452,6851,1) AS layer1) AS a,
	     (SELECT generate_series(6448,6451,1) AS layer2) AS b
	ORDER BY random_seed LIMIT 100
) RETURNING config_id;--7609-7708

BEGIN;
DELETE FROM hps3.config_mlp_sgd 
USING stochastic.layer_conditional1 AS a
WHERE layer_array[1] = a.layer_id AND start_time IS NULL
AND (a.W_lr_scale[1] = 0.001::FLOAT4 OR a.W_lr_scale[2] = 0.001::FLOAT4 OR a.W_lr_scale[3] = 0.001::FLOAT4) AND task_id = 24;
COMMIT;

UPDATE hps3.config_mlp_sgd SET description = 'stoch2b no NaNs? new channels' WHERE task_id = 23 AND start_time IS NULL

SELECT COUNT(*) FROM hps3.config_mlp_sgd WHERE task_id = 24 AND start_time IS NULL;