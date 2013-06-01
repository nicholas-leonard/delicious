/** 200 40 .1 **/


/* Baseline */

INSERT INTO stochastic.layer_conditional5 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,noise_beta,noise_stdev,noise_normality,noise_scale,max_col_norm,irange) (
SELECT 'conditional5', dim, dim*hdp,0,0,NULL,noise_stdev,1.0,1.0,ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{20}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0,1.0,7.0}'::FLOAT4[]) AS noise_stdev) AS c,
	(SELECT unnest('{NULL,2.1,1}'::FLOAT4[]) AS mcn) AS d
)RETURNING layer_id;--8190-8198

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',59,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,lr,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Baseline'
	FROM (SELECT generate_series(8190,8198,1) AS layer1) AS a,
		(SELECT unnest('{0.1}'::FLOAT4[]) AS lr) AS b
) RETURNING config_id;--9256-9264

/* Noisy Rectifier */

INSERT INTO stochastic.layer_conditional4 (
layer_class,dim,hidden_dim,gater_activation,sparsity_cost_coeff,noise_stdev,
sparsity_target,sparsity_decay,sparse_init,W_lr_scale,b_lr_scale,max_col_norm,irange) (
SELECT  'conditional4', dim, (dim*hdp)::INT4,'rectifiedlinear',sparsity_cost_coeff,ns,st,sd,
	ARRAY[NULL,NULL,15],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],ARRAY[mcn,mcn,mcn],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) + (hdp*dim))), NULL]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS c,
	(SELECT unnest('{0.0,0.5,1.0}'::FLOAT4[]) AS ns ) AS d, 
	(SELECT unnest('{0.00001}'::FLOAT4[]) AS sparsity_cost_coeff) AS e,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS f,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS sd) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g
)RETURNING layer_id;--8115-8126

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',56,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL,random()*1000000 AS random_seed, 'Noisy Rectifier 200 40 .1'
	FROM (SELECT generate_series(8115,8126,1) AS layer1) AS a
) RETURNING config_id;--9163-9174

/* Smooth Times Stochastic */

INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,noise_beta,0,noise_normality,ARRAY[cn,cn,cn]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{40.1,2.6}'::FLOAT4[]) AS noise_beta) AS d,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS cn) AS e
)RETURNING layer_id;--8238-8239

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',65,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Smooth Time Stochastic'
	FROM (SELECT generate_series(8238,8239,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--9316-9317


INSERT INTO stochastic.layer_conditional3 (
layer_class,dim,hidden_dim,sparsity_target,sparsity_cost_coeff,stochastic_ratio,noise_beta,
noise_stdev,noise_normality,max_col_norm,irange) (
SELECT 'conditional3', dim, dim*hdp,0.1,0.001,0.5,NULL,noise_stdev,noise_normality,ARRAY[cn,cn,cn]::FLOAT4[],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{200}'::FLOAT4[]) AS dim) AS a,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS noise_normality) AS c,
	(SELECT unnest('{0,0.1,0.5,1.0,2.0}'::FLOAT4[]) AS noise_stdev) AS d,
	(SELECT unnest('{NULL,1,2.1}'::FLOAT4[]) AS cn) AS e
)RETURNING layer_id;--8127-8141


INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',57,18,4,'{1}'::INT8[],
		'{1,7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,0.5::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Smooth Time Stochastic'
	FROM (SELECT generate_series(8127,8141,1) AS layer1) AS a
	UNION ALL
	SELECT 'mlp','sgd',57,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1, 7572]::INT8[],'{9}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Smooth Time Stochastic'
	FROM (SELECT generate_series(8127,8141,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--9199-9228


/* Stochastic Binary Neuron */

INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,st,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.1,0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--8229-8230

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',63,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Stochastic Binary Neuron'
	FROM (SELECT generate_series(8229,8230,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--9307-9308


INSERT INTO stochastic.layer_stochastic2 (
layer_class,hidden_activation,dim,hidden_dim,stoch_grad_coeff,mean_loss_coeff,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic2','tanh',dim,(dim*hdp)::INT4,sgc,mean_loss_coeff,st,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,0.01,0.01],ARRAY[lrs,0.01,0.01],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.9}'::FLOAT4[]) AS mean_loss_coeff) AS c,
	(SELECT unnest('{3}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1,1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS sgc) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--8233-8234

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',63,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Stochastic Binary Neuron'
	FROM (SELECT generate_series(8233,8234,1) AS layer1) AS a
	ORDER BY random_seed
) RETURNING config_id;--9311-9312

SELECT task_id, COUNT(*) FROM hps3.config_mlp_sgd WHERE task_id > 62 AND end_time IS NULL GROUP BY task_id
SELECT task_id, COUNT(*) FROM hps3.config_mlp_sgd WHERE task_id > 55 AND start_time IS NULL GROUP BY task_id

SELECT * FROM hps3.config_mlp_sgd WHERE task_id = 64 AND start_time IS NOT NULL

UPDATE hps3.config_mlp_sgd SET start_time = NULL WHERE config_id = 9167


/* Straight Through */

INSERT INTO stochastic.layer_stochastic4 (
layer_class,hidden_activation,dim,hidden_dim,derive_sigmoid,sparsity_target,sparsity_cost_coeff,max_col_norm,irange,w_lr_scale,b_lr_scale) (
SELECT 'stochastic4','tanh',dim,(dim*hdp)::INT4,ds,st,scc,ARRAY[mcn,mcn,mcn],ARRAY[lrs,lrs,lrs],ARRAY[lrs,lrs,lrs],
	ARRAY[sqrt(6. / ((28*28) + dim)), sqrt(6. / ((28*28) +(dim*hdp))), 4*sqrt(6. / ((dim*hdp) + dim))]::FLOAT4[]
FROM (SELECT unnest('{2000}'::INT4[]) AS dim) AS a,
	(SELECT unnest('{0.2}'::FLOAT4[]) AS hdp) AS b,
	(SELECT unnest('{0.1}'::FLOAT4[]) AS scc) AS e,
	(SELECT unnest('{2.1}'::FLOAT4[]) AS mcn) AS f,
	(SELECT unnest('{True}'::BOOLEAN[]) AS ds) AS h,
	(SELECT unnest('{1.0}'::FLOAT4[]) AS lrs) AS g,
	(SELECT unnest('{0.01}'::FLOAT4[]) AS st) AS i
)RETURNING layer_id;--8231-8232

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,channel_array,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate,init_momentum,random_seed, description) (
	SELECT 'mlp','sgd',64,18,4,'{1}'::INT8[],
		'{7}'::INT8[],'{5,6}'::INT8[],ARRAY[layer1, 7764]::INT8[],'{8}'::INT8[],
		32,0.1,NULL::FLOAT4 AS init_momentum,random()*1000000 AS random_seed, 'Straight-through'
	FROM (SELECT generate_series(8237,8237,1) AS layer1) AS a
) RETURNING config_id;--9309-9310...9315


UPDATE hps3.config_mlp_sgd SET task_id = 64 WHERE config_id BETWEEN 9309 AND 9310

python plot_curves_multi_model.py 9075 'Noisy Rectifier' 9309 Straight-through 9307 'Stochastic Binary Neuron' 8747 'Smooth Times Stochastic' 0.015 0.04