CREATE SCHEMA stochastic;

CREATE TABLE stochastic.layer_stochastic1 (
	dim			INT4,
	hidden_dim		INT4,
	mean_loss_coeff 	FLOAT4 DEFAULT 0.5,
	hidden_activation 	VARCHAR(255) DEFAULT 'tanh',
	sparsity_target 	FLOAT4 DEFAULT 0.1,
	sparsity_cost_coeff 	FLOAT4 DEFAULT 1.0,
	irange 			FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	istdev 			FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	sparse_init 		INT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::INT4[],
	sparse_stdev 		FLOAT4[3] DEFAULT ARRAY[1.,1.,1.]::FLOAT4[],
	init_bias 		FLOAT4[3] DEFAULT ARRAY[0.,0.,0.]::FLOAT4[],
	W_lr_scale 		FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	b_lr_scale 		FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	max_col_norm		FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	weight_decay_coeff	FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	PRIMARY KEY (layer_id)
) INHERITS (hps3.layer);


CREATE TABLE stochastic.layer_stochastic2 (
	dim			INT4,
	hidden_dim		INT4,
	mean_loss_coeff 	FLOAT4 DEFAULT 0.5,
	hidden_activation 	VARCHAR(255) DEFAULT 'tanh',
	sparsity_target 	FLOAT4 DEFAULT 0.1,
	sparsity_cost_coeff 	FLOAT4 DEFAULT 1.0,
	irange 			FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	istdev 			FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	sparse_init 		INT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::INT4[],
	sparse_stdev 		FLOAT4[3] DEFAULT ARRAY[1.,1.,1.]::FLOAT4[],
	init_bias 		FLOAT4[3] DEFAULT ARRAY[0.,0.,0.]::FLOAT4[],
	W_lr_scale 		FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	b_lr_scale 		FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	max_col_norm		FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	weight_decay_coeff	FLOAT4[3] DEFAULT ARRAY[NULL,NULL,NULL]::FLOAT4[],
	PRIMARY KEY (layer_id)
) INHERITS (hps3.layer);

CREATE TABLE stochastic.layer_stochasticsoftmax (
	weight_decay_coeff	FLOAT4,
	PRIMARY KEY (layer_id);
) INHERITS (hps3.layer_softmax);

CREATE TABLE stochastic.cost_stochastic1 () INHERITS (hps3.cost_mlp);

CREATE TABLE stochastic.ddm_newsgroups20 (
	which_set	VARCHAR(255), 
	data_path	VARCHAR(255) DEFAULT NULL, 
	valid_ratio	FLOAT4 DEFAULT 0.2,
        sum_to_one	BOOLEAN DEFAULT True,
	one_hot 	BOOLEAN DEFAULT True,
	PRIMARY KEY (ddm_id)
) INHERITS (hps3.ddm);