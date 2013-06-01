SELECT 	a.config_id, a.dataset_id, task_id, a.description, b.epoch_count AS epoch, hps3.get_end(a.config_id) AS eend, (a.end_time IS NOT NULL) AS done,
	a.learning_rate, a.init_momentum, v.dim, v.hidden_dim, v.stochastic_ratio, v.sparsity_target, v.noise_beta, v.noise_stdev, v.noise_normality, v.max_col_norm, a.layer_array, a.ext_array, 
	w.dim, w.hidden_dim, w.stochastic_ratio, w.sparsity_target, w.noise_beta, w.noise_stdev, w.noise_normality, w.max_col_norm, 
	hps3.get_channel(a.config_id, 'train_conditional30_mean_output_sparsity', b.epoch_count) AS mean_sparsity, 
	hps3.get_channel(a.config_id, 'train_conditional30_max_unit_sparsity_prop', b.epoch_count) AS maxusp,
	hps3.get_channel(a.config_id, 'train_conditional30_min_unit_sparsity_prop', b.epoch_count) AS minusp,
	hps3.get_channel(a.config_id, 'train_conditional30_mean_sparsity_prop0.2', b.epoch_count) AS msp2,
	hps3.get_channel(a.config_id, 'train_conditional30_mean_sparsity_prop0.3', b.epoch_count) AS msp3,
	hps3.get_channel(a.config_id, 'train_conditional30_mean_sparsity_prop0.4', b.epoch_count) AS msp4,
	hps3.get_channel(a.config_id, 'train_conditional30_mean_sparsity_prop', b.epoch_count) AS msp5,
	hps3.get_channel(a.config_id, 'train_conditional30_mean_unit_sparsity_meta_prop', b.epoch_count) AS musmp,
	hps3.get_channel(a.config_id, 'train_conditional30_mean_unit_sparsity_meta_prop2', b.epoch_count) AS musmp2,
	hps3.get_channel(a.config_id, 'train_conditional30_output_stdev', b.epoch_count) AS stdev,
	hps3.get_channel(a.config_id, 'train_conditional30_output_meta_stdev', b.epoch_count) AS mstdev,
	hps3.get_channel(a.config_id, 'train_conditional31_mean_output_sparsity', b.epoch_count) AS mean_sparsity, 
	hps3.get_channel(a.config_id, 'train_conditional31_max_unit_sparsity_prop', b.epoch_count) AS maxusp,
	hps3.get_channel(a.config_id, 'train_conditional31_min_unit_sparsity_prop', b.epoch_count) AS minusp,
	hps3.get_channel(a.config_id, 'train_conditional31_mean_sparsity_prop0.2', b.epoch_count) AS msp2,
	hps3.get_channel(a.config_id, 'train_conditional31_mean_sparsity_prop0.3', b.epoch_count) AS msp3,
	hps3.get_channel(a.config_id, 'train_conditional31_mean_sparsity_prop0.4', b.epoch_count) AS msp4,
	hps3.get_channel(a.config_id, 'train_conditional31_mean_sparsity_prop', b.epoch_count) AS msp5,
	hps3.get_channel(a.config_id, 'train_conditional31_mean_unit_sparsity_meta_prop', b.epoch_count) AS musmp,
	hps3.get_channel(a.config_id, 'train_conditional31_mean_unit_sparsity_meta_prop2', b.epoch_count) AS musmp2,
	hps3.get_channel(a.config_id, 'train_conditional31_output_stdev', b.epoch_count) AS stdev,
	hps3.get_channel(a.config_id, 'train_conditional31_output_meta_stdev', b.epoch_count) AS mstdev,
	b.channel_value AS optimum_valid_mca, 
	hps3.get_channel(a.config_id, 'train_objective', b.epoch_count) AS optimum_train_error, 
	hps3.get_channel(a.config_id, 'test_hps_cost', b.epoch_count) AS optimum_test_err ,
	hps3.get_last(a.config_id, 'valid_hps_cost') AS final_valid_err, hps3.get_last(a.config_id, 'train_objective') AS final_train_error, 
	--a.dropout_include_probs, a.dropout_input_include_prob, a.dropout_input_scale, a.weight_decay, 
	a.term_array, a.worker_name
FROM 	hps3.config_mlp_sgd AS a, (	
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value ASC, epoch_count ASC)
		FROM hps3.training_log
		WHERE channel_name = 'valid_hps_cost'
	) AS b, hps3.dataset AS p, stochastic.layer_conditional3 AS v, stochastic.layer_conditional3 AS w, hps3.layer_softmax AS n
WHERE  a.config_id = b.config_id AND b.rank = 1 AND (task_id = 41) AND a.dataset_id = p.dataset_id 
	AND a.layer_array[1] = v.layer_id AND a.layer_array[1] = w.layer_id 
	AND (a.layer_array[3] = n.layer_id OR a.layer_array[4] = n.layer_id OR a.layer_array[5] = n.layer_id OR a.layer_array[6] = n.layer_id)
	--AND m.channel_value < 0.25 AND b.channel_value > 0.1 AND j.epoch_count > 5
ORDER BY b.channel_value ASC
--/(hps3.get_channel(a.config_id, 'train_conditional20_mean_output_sparsity', b.epoch_count) ^2)) DESC


/*
SELECT * FROM hps3.config_mlp_sgd WHERE config_id =  5427
0;2.30162;0.8738;0.868399;0.999891
1;0.488699;0.139;0.1485;0.0478681
2;0.454176;0.1292;0.1316;0.0406478
3;0.378996;0.106;0.109;0.0480657
4;0.410552;0.1118;0.1142;0.0501002
5;0.409672;0.1205;0.1284;0.0514651
6;0.362928;0.0949001;0.1025;0.0597128
7;0.363698;0.1009;0.1029;0.0611203
8;0.298865;0.0816001;0.0868001;0.0669479
9;0.33842;0.0787001;0.0911001;0.0771834

SELECT a.epoch_count, a.channel_value, b.channel_value, c.channel_value, d.channel_value 
FROM hps3.training_log AS a, hps3.training_log AS b, hps3.training_log AS c, hps3.training_log AS d
WHERE a.config_id = 8841 AND (b.config_id, b.epoch_count) = (a.config_id, a.epoch_count)
	AND (c.config_id, c.epoch_count) = (a.config_id, a.epoch_count) AND (d.config_id, d.epoch_count) = (a.config_id, a.epoch_count) 
	AND a.channel_name = 'train_objective' AND b.channel_name = 'valid_hps_cost' AND c.channel_name = 'test_hps_cost' AND d.channel_name = 'train_conditional30_mean_sparsity_prop0.2'
ORDER BY epoch_count ASC

SELECT * 
FROM hps3.training_log
WHERE channel_name = 'train_stochastic10_mean_output_sparsity'
ORDER BY channel_value ASC LIMIT 10

SELECT weight_decay_coeff[1] AS h, MIN(validity), AVG(validity), MAX(validity) 
FROM	(
	SELECT 	a.config_id, a.description, b.epoch_count AS epoch, j.epoch_count AS eend, (a.end_time IS NOT NULL) AS done,
		n.w_lr_scale, v.dim, v.hidden_dim, v.mean_loss_coeff, v.sparsity_target, v.sparsity_cost_coeff, 
		v.weight_decay_coeff, v.max_col_norm, v.W_lr_scale AS wlrs, (b.channel_value/(m.channel_value^2)) AS validity,
		a.layer_array, a.ext_array, m.channel_value AS optimum_sparsity, a.init_momentum,
		b.channel_value AS optimum_valid_mca, j.channel_value AS final_valid_mca, 
		k.channel_value AS optimum_train_error, l.channel_value AS final_train_error, 
		--a.dropout_include_probs, a.dropout_input_include_prob, a.dropout_input_scale, a.weight_decay, 
		a.term_array, a.worker_name
	FROM 	hps3.config_mlp_sgd AS a,
		(
			SELECT config_id, epoch_count, channel_value, rank()
				OVER (PARTITION BY config_id ORDER BY channel_value DESC, epoch_count ASC)
			FROM hps3.training_log
			WHERE channel_name = 'valid_hps_mca'
		) AS b, 
		(
			SELECT config_id, epoch_count, channel_value, rank()
				OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
			FROM hps3.training_log
			WHERE channel_name = 'valid_hps_mca'
		) AS j, 
		(
			SELECT config_id, epoch_count, channel_value, rank()
				OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
			FROM hps3.training_log
			WHERE channel_name = 'train_objective'
		) AS l, hps3.training_log AS k, hps3.dataset AS p, stochastic.layer_stochastic2 AS v, hps3.training_log AS m, stochastic.layer_stochasticsoftmax AS n
	WHERE  a.config_id = b.config_id AND b.rank = 1
		AND a.config_id = k.config_id AND k.channel_name = 'train_objective' AND k.epoch_count = b.epoch_count 
		AND a.config_id = m.config_id AND m.channel_name = 'train_stochastic20_mean_output_sparsity' AND m.epoch_count = b.epoch_count 
		AND a.config_id = j.config_id AND j.rank = 1 AND a.config_id = l.config_id AND l.rank = 1
		AND a.task_id = 23 AND a.dataset_id = p.dataset_id AND a.layer_array[1] = v.layer_id AND a.layer_array[2] = n.layer_id
		--AND m.channel_value < 0.25 AND b.channel_value > 0.1 AND j.epoch_count > 5
	ORDER BY validity DESC
	) AS a
GROUP BY h
ORDER BY avg DESC


dim: 80-160    
hidden_dim: 32   propotion: 0.2
wlrscales: 0.01,0.01-0.001,0.1-0.01, "{0.01-0.1,0.001-0.01,0.01}"
wdecaycoeff: 
sparsity_cost_coeff: 10

*/