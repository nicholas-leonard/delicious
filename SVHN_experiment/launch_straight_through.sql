/* Straight Through 

Try to bypass expert activation...*/

INSERT INTO stochastic.layer_straightthrough (
layer_class,hidden_activation,expert_activation,dim,hidden_dim,derive_sigmoid,sparsity_target,sparsity_cost_coeff,
max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'straightthrough','tanh',ea,dim,(dim*hdp)::INT4,False,st,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],
	ARRAY[4*sqrt(6. / ((32*32) + dim)), sqrt(6. / ((32*32) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{256,1024,4096}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.0001,0.01,1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest(ARRAY['tanh',NULL]::VARCHAR[]) AS ea) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--9371-9388

UPDATE stochastic.layer_straightthrough
SET layer_channel_names = ARRAY['max_unit_sparsity_prop', 'mean_output_sparsity',
	'mean_sparsity_prop', 'min_unit_sparsity_prop', 'mean_unit_sparsity_meta_prop',
        'mean_unit_sparsity_meta_prop2','mean_sparsity_prop0.2','mean_sparsity_prop0.3',
        'mean_sparsity_prop0.4','output_stdev','output_meta_stdev', 'stoch_grad',
        'kl_grad' ]::VARCHAR(255)[]
WHERE layer_id BETWEEN 9371 AND 9388


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',88,60,5,'{1}'::INT8[],
		'{7}'::INT8[],'{5,4}'::INT8[],ARRAY[layer1, 9389]::INT8[],'{8}'::INT8[],
		64,lr,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'ST-baseline'
	FROM (SELECT generate_series(9371,9388,1) AS layer1) AS a,
		(SELECT unnest('{0.1,0.01}'::FLOAT4[]) AS lr) AS i
	ORDER BY random_seed
) RETURNING config_id;--11234-11269

UPDATE hps3.config_mlp_sgd SET input_space_id = NULL WHERE task_id = 88

SELECT task_id, COUNT(*) FROM hps3.config_mlp_sgd WHERE task_id > 55 GROUP BY task_id
SELECT MAX(task_id) FROM hps3.config_mlp_sgd 

UPDATE hps3.config_mlp_sgd SET start_time = NULL WHERE config_id = 9180