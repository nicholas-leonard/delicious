SELECT 	a.config_id, a.description, b.epoch_count AS epoch, j.epoch_count AS eend, (a.end_time IS NOT NULL) AS done,
	n.w_lr_scale, v.dim, v.hidden_dim, v.mean_loss_coeff, v.sparsity_target, v.sparsity_cost_coeff, 
	v.weight_decay_coeff, v.max_col_norm, v.W_lr_scale,
	a.layer_array, a.ext_array, m.channel_value AS optimum_sparsity, a.init_momentum,
	b.channel_value AS optimum_valid_mca, k.channel_value AS optimum_train_error, o.channel_value AS optimum_test_mca ,
	j.channel_value AS final_valid_mca, l.channel_value AS final_train_error, 
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
	) AS l, hps3.training_log AS k, hps3.dataset AS p, stochastic.layer_stochastic2 AS v, hps3.training_log AS m, 
		stochastic.layer_stochasticsoftmax AS n, hps3.training_log AS o
WHERE  a.config_id = b.config_id AND b.rank = 1
	AND a.config_id = k.config_id AND k.channel_name = 'train_objective' AND k.epoch_count = b.epoch_count 
	AND a.config_id = m.config_id AND m.channel_name = 'train_stochastic20_mean_output_sparsity' AND m.epoch_count = b.epoch_count  
	AND a.config_id = o.config_id AND o.channel_name = 'test_hps_mca' AND o.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1 AND a.config_id = l.config_id AND l.rank = 1
	AND a.task_id = 23 AND a.dataset_id = p.dataset_id AND a.layer_array[1] = v.layer_id AND a.layer_array[2] = n.layer_id
	--AND m.channel_value < 0.25 AND b.channel_value > 0.1 AND j.epoch_count > 5
ORDER BY (b.channel_value/(m.channel_value^2)) DESC


/*
SELECT * FROM hps3.config_mlp_sgd WHERE config_id =  5427

SELECT a.epoch_count, a.channel_value, b.channel_value, c.channel_value, d.channel_value
FROM hps3.training_log AS a, hps3.training_log AS b, hps3.training_log AS c, hps3.training_log AS d
WHERE a.config_id = 7549 AND (b.config_id, b.epoch_count) = (a.config_id, a.epoch_count)
	AND (c.config_id, c.epoch_count) = (a.config_id, a.epoch_count) AND (d.config_id, d.epoch_count) = (a.config_id, a.epoch_count) 
	AND a.channel_name = 'train_objective' AND b.channel_name = 'valid_hps_mca' AND c.channel_name = 'test_hps_mca' AND d.channel_name = 'train_stochastic20_mean_output_sparsity'
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

SELECT * FROM hps3.training_log WHERE config_id = 332 AND channel_name = 'train_objective'
ORDER BY epoch_count DESC

SELECT 	a.config_id, a.description, b.epoch_count AS epoch, j.epoch_count AS end, a.dataset_id,
	v.dim, a.cost_array, a.layer_array, a.ext_array, p.preprocess_array,
	b.channel_value AS optimum_valid_nll, j.channel_value AS final_valid_nll, 
	k.channel_value AS optimum_train_error, l.channel_value AS final_train_error, 
	a.init_momentum, a.learning_rate, a.batch_size, 
	--a.dropout_include_probs, a.dropout_input_include_prob, a.dropout_input_scale, a.weight_decay, 
	a.term_array, a.worker_name, a.start_time, a.end_time
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
	) AS l, hps3.training_log AS k, hps3.dataset AS p, hps3.layer_tanh AS v
WHERE  a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_objective' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1 AND a.config_id = l.config_id AND l.rank = 1
	AND a.task_id = 19 AND a.dataset_id = p.dataset_id AND a.layer_array[1] = v.layer_id
ORDER BY optimum_valid_nll DESC

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

CREATE OR REPLACE FUNCTION hps3.get_best(config_id INT4, channel_name VARCHAR(1000)) 
RETURNS TABLE (epoch INT4, value FLOAT4, rank INT4) AS $$
	SELECT epoch_count, channel_value, rank::INT4
	FROM	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value DESC, epoch_count ASC)
		FROM hps3.training_log
		WHERE channel_name = $2 AND config_id = $1
		) AS a
$$ LANGUAGE SQL;

*/