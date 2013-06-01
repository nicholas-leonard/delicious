/* Pure Stochastic */

INSERT INTO stochastic.layer_stochastic3 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic3','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,st,scc,ARRAY[mcn,mcn*10],ARRAY[0.01,0.01],ARRAY[0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{100,200,400}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{2,4,8}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{0.05,0.1,0.5}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--7841-7867

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',45,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch3a'
	FROM (SELECT generate_series(7841,7867,1) AS layer1) AS a
) RETURNING config_id;--8898-8924

/* Dept */

INSERT INTO stochastic.layer_stochastic3 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic3','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,st,scc,ARRAY[mcn,mcn*10],ARRAY[0.01,0.01],ARRAY[0.01,0.01],
	ARRAY[sqrt(6. / ((dim) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{100,200,400}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{2,4,8}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{0.05,0.1,0.5}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--7895-7921

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',45,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1,layer2,7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch3ba dept+1'
	FROM (SELECT generate_series(7841,7867,1) AS layer1, generate_series(7868,7894,1) AS layer2) AS a
) RETURNING config_id;--8952-8978
--DELETE FROM hps3.config_mlp_sgd WHERE config_id BETWEEN 8925 AND 8951


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',45,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1,layer2,layer2,7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch3bb dept+2'
	FROM (SELECT generate_series(7841,7867,1) AS layer1, generate_series(7868,7894,1) AS layer2) AS a
) RETURNING config_id;--8979-9005


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',45,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1,layer2,layer2,layer2,layer2,7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'stoch3bc dept+4'
	FROM (SELECT generate_series(7841,7867,1) AS layer1, generate_series(7868,7894,1) AS layer2) AS a
) RETURNING config_id;--9006-9032


UPDATE hps3.config_mlp_sgd SET task_id = 47 WHERE config_id BETWEEN 8952 AND 9032

UPDATE hps3.config_mlp_sgd SET start_time = NULL WHERE config_id = 8918
