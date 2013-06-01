SELECT 	a.config_id, a.dataset_id, a.description, b.epoch_count AS epoch, hps3.get_end(a.config_id) AS eend, (a.end_time IS NOT NULL) AS done,
	n.w_lr_scale, v.hidden_activation, v.W_lr_scale, v.max_col_norm, v.dim, v.hidden_dim, v.derive_sigmoid, v.sparsity_target, v.sparsity_cost_coeff, 
	a.layer_array, a.ext_array, 
	hps3.get_channel(a.config_id, 'train_stochastic40_mean_output_sparsity', b.epoch_count) AS mean_sparsity, 
	hps3.get_channel(a.config_id, 'train_stochastic40_max_unit_sparsity_prop', b.epoch_count) AS maxusp,
	hps3.get_channel(a.config_id, 'train_stochastic40_min_unit_sparsity_prop', b.epoch_count) AS minusp,
	hps3.get_channel(a.config_id, 'train_stochastic40_mean_sparsity_prop0.2', b.epoch_count) AS msp2,
	hps3.get_channel(a.config_id, 'train_stochastic40_mean_sparsity_prop0.3', b.epoch_count) AS msp3,
	hps3.get_channel(a.config_id, 'train_stochastic40_mean_sparsity_prop0.4', b.epoch_count) AS msp4,
	hps3.get_channel(a.config_id, 'train_stochastic40_mean_sparsity_prop', b.epoch_count) AS msp5,
	hps3.get_channel(a.config_id, 'train_stochastic40_mean_unit_sparsity_meta_prop', b.epoch_count) AS musmp,
	hps3.get_channel(a.config_id, 'train_stochastic40_mean_unit_sparsity_meta_prop2', b.epoch_count) AS musmp2,
	b.channel_value AS optimum_valid_mca, 
	hps3.get_channel(a.config_id, 'train_objective', b.epoch_count) AS optimum_train_error, 
	hps3.get_channel(a.config_id, 'test_hps_cost', b.epoch_count) AS optimum_test_mca ,
	hps3.get_last(a.config_id, 'valid_hps_cost') AS final_valid_mca, hps3.get_last(a.config_id, 'train_objective') AS final_train_error, 
	--a.dropout_include_probs, a.dropout_input_include_prob, a.dropout_input_scale, a.weight_decay, 
	a.term_array, a.worker_name
FROM 	hps3.config_mlp_sgd AS a, (	
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value ASC, epoch_count ASC)
		FROM hps3.training_log
		WHERE channel_name = 'valid_hps_cost'
	) AS b, hps3.dataset AS p, stochastic.layer_stochastic4 AS v, stochastic.layer_stochasticsoftmax AS n
WHERE  a.config_id = b.config_id AND b.rank = 1 AND (a.task_id = 60) AND a.dataset_id = p.dataset_id 
	AND a.layer_array[1] = v.layer_id AND a.layer_array[2] = n.layer_id AND hps3.get_channel(a.config_id, 'train_stochastic40_mean_output_sparsity', b.epoch_count) < 0.14


/*
SELECT * FROM hps3.config_mlp_sgd WHERE config_id =  5427

SELECT a.epoch_count, a.channel_value, b.channel_value, c.channel_value, d.channel_value, e.channel_value
FROM hps3.training_log AS a, hps3.training_log AS b, hps3.training_log AS c, hps3.training_log AS d, hps3.training_log AS e
WHERE a.config_id = 8872 AND (b.config_id, b.epoch_count) = (a.config_id, a.epoch_count)
	AND (c.config_id, c.epoch_count) = (a.config_id, a.epoch_count) AND (d.config_id, d.epoch_count) = (a.config_id, a.epoch_count) 
	AND a.channel_name = 'train_objective' AND b.channel_name = 'valid_hps_mca' 
	AND c.channel_name = 'test_hps_mca' AND d.channel_name = 'train_stochastic20_mean_output_sparsity'
	AND e.channel_name = 'train_stochastic20_max_unit_sparsity_prop' AND (e.config_id, e.epoch_count) = (a.config_id, a.epoch_count) 
ORDER BY epoch_count ASC


SELECT 	a.epoch_count, a.channel_value AS train_cost,
	hps3.get_channel(a.config_id::INT4, 'valid_hps_cost'::VARCHAR, a.epoch_count) AS valid_error, 
	hps3.get_channel(a.config_id::INT4, 'test_hps_cost'::VARCHAR, a.epoch_count) AS test_error
FROM hps3.training_log AS a
WHERE a.config_id = 8872 AND a.channel_name = 'train_objective'
ORDER BY epoch_count ASC

SELECT a.epoch_count, a.channel_value, b.channel_value, c.channel_value
FROM hps3.training_log AS a, hps3.training_log AS b, hps3.training_log AS c
WHERE a.config_id = 5427 AND (b.config_id, b.epoch_count) = (a.config_id, a.epoch_count)
	AND (c.config_id, c.epoch_count) = (a.config_id, a.epoch_count) 
	AND a.channel_name = 'train_objective' AND b.channel_name = 'valid_hps_cost' AND c.channel_name = 'test_hps_cost'
ORDER BY epoch_count ASC

SELECT * 
FROM hps3.training_log
WHERE channel_name = 'train_stochastic10_mean_output_sparsity'
ORDER BY channel_value ASC LIMIT 10

SELECT a.epoch_count, a.channel_value, d.channel_value, b.channel_value, e.channel_value, c.channel_value, f.channel_value
FROM hps3.training_log AS a, hps3.training_log AS b, hps3.training_log AS c, hps3.training_log AS d, hps3.training_log AS e, hps3.training_log AS f
WHERE a.config_id = 332 AND d.config_id = 366 AND (b.config_id, b.epoch_count) = (a.config_id, a.epoch_count)
	AND (c.config_id, c.epoch_count) = (a.config_id, a.epoch_count) AND d.epoch_count = a.epoch_count
	AND (e.config_id, e.epoch_count) = (d.config_id, d.epoch_count) AND (f.config_id, f.epoch_count) = (d.config_id, d.epoch_count) 
	AND a.channel_name = 'train_objective' AND b.channel_name = 'valid_hps_mca' AND c.channel_name = 'test_hps_mca'
	AND d.channel_name = 'train_objective' AND e.channel_name = 'valid_hps_mca' AND f.channel_name = 'test_hps_mca'
ORDER BY a.epoch_count ASC

*/