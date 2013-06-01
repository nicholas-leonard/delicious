
/* Straight Through */

INSERT INTO stochastic.layer_stochastic4 (
layer_class,hidden_activation,dim,hidden_dim,derive_sigmoid,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic4','tanh',dim,(dim*hdp)::INT4,ds,st,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{200}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.0001,0.01,1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{True,False}'::BOOLEAN[]) AS ds) AS h,
	(SELECT unnest('{1.0,0.1,0.01}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--8199-8216

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',60,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Straight Through'
	FROM (SELECT generate_series(8199,8216,1) AS layer1) AS a
	UNION ALL
	SELECT 'mlp','sgd',60,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Straight Through'
	FROM (SELECT generate_series(8199,8216,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--9265-9300

SELECT task_id, COUNT(*) FROM hps3.config_mlp_sgd WHERE task_id > 55 GROUP BY task_id
SELECT MAX(task_id) FROM hps3.config_mlp_sgd 

UPDATE hps3.config_mlp_sgd SET start_time = NULL WHERE config_id = 9180