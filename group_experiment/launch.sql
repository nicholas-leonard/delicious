/* Grouped Straight Through */

INSERT INTO stochastic.layer_group1 (
layer_class,hidden_activation,expert_activation,gater_dim,hidden_dim,expert_dim,derive_sigmoid,
sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'group1','tanh',ea,gater_dim,hidden_dim,expert_dim,ds,st,scc,
	ARRAY[mcn,mcn,mcn],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],
	ARRAY[sqrt(6. / ((28*28) + (gater_dim*expert_dim))), sqrt(6. / ((28*28) + hidden_dim)), 4*sqrt(6. / (hidden_dim + (gater_dim*expert_dim)))]::FLOAT4[]
FROM (SELECT unnest('{32,64,128}'::INT4[]) AS gater_dim) AS a,
	(SELECT unnest('{100,200,400}'::INT4[]) AS hidden_dim) AS b,
	(SELECT unnest('{4,8,16,32}'::INT4[]) AS expert_dim) AS c, 
	(SELECT unnest('{0.1,1.0}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{True,False}'::BOOLEAN[]) AS ds) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS i,
	(SELECT unnest( ARRAY['tanh',NULL]::VARCHAR[]) AS ea) AS j
)RETURNING layer_id;--8261-8548

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',66,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Group ST'
	FROM (SELECT generate_series(8261,8548,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--9339-9626

/* Overlapping Random Groups ST */

INSERT INTO stochastic.layer_group2 (
layer_class,hidden_activation,expert_activation,dim,gater_dim,hidden_dim,group_prob,derive_sigmoid,
sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'group2','tanh',ea,dim,gater_dim,hidden_dim,group_prob,ds,st,scc,
	ARRAY[mcn,mcn,mcn],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + hidden_dim)), 4*sqrt(6. / (hidden_dim + gater_dim))]::FLOAT4[]
FROM (SELECT unnest('{512,1024,2048}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{64,128,256}'::INT4[]) AS hidden_dim) AS b,
	(SELECT unnest('{128,512,1024}'::INT4[]) AS gater_dim) AS c, 
	(SELECT unnest('{0.01,0.001}'::FLOAT4[]) AS group_prob) AS k,
	(SELECT unnest('{0.1,1.0}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{False}'::BOOLEAN[]) AS ds) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS i,
	(SELECT unnest( ARRAY[NULL]::VARCHAR[]) AS ea) AS j
WHERE (gater_dim*dim*group_prob)/(gater_dim+dim) >= 1
)RETURNING layer_id;--8727-8780

SELECT gater_dim,dim,group_prob,(gater_dim*dim*group_prob)/(gater_dim+hidden_dim) FROM stochastic.layer_group2 WHERE layer_id BETWEEN 8657 AND 8726

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',67,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'ORG-ST'
	FROM (SELECT generate_series(8727,8780,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--9805-9858

UPDATE hps3.config_mlp_sgd SET start_time = NULL, end_time = NULL WHERE task_id = 67

/* misc commands */

BEGIN;
UPDATE hps3.config_mlp_sgd AS a SET start_time = NULL 
WHERE (SELECT b.config_id FROM hps3.training_log AS b WHERE b.config_id = a.config_id LIMIT 1) IS NULL 
AND task_id = 66;
COMMIT;

SELECT * FROM hps3.config_mlp_sgd WHERE task_id = 66;