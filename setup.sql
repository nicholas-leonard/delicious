CREATE SCHEMA stochastic;

CREATE TABLE stochastic.layer_group1 (
	gater_dim		INT4,
	hidden_dim		INT4,
	expert_dim		INT4,
	expert_activation 	VARCHAR(255),
	derive_sigmoid 		BOOLEAN DEFAULT False,
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


CREATE TABLE stochastic.layer_group2 (
	dim			INT4,
	gater_dim		INT4,
	hidden_dim		INT4,
	group_prob		FLOAT4,
	expert_activation 	VARCHAR(255),
	derive_sigmoid 		BOOLEAN DEFAULT False,
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

CREATE TABLE stochastic.layer_StochasticBinaryNeuron (
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

--DROP TABLE stochastic.layer_GaterOnly;
CREATE TABLE stochastic.layer_GaterOnly (
	dim			INT4,
	hidden_dim		INT4,
	mean_loss_coeff 	FLOAT4 DEFAULT 0.9,
	hidden_activation 	VARCHAR(255) DEFAULT 'tanh',
	sparsity_target 	FLOAT4 DEFAULT 0.1,
	sparsity_cost_coeff 	FLOAT4 DEFAULT 1.0,
	stoch_grad_coeff	FLOAT4 DEFAULT 1.0,
	irange 			FLOAT4[2] DEFAULT ARRAY[NULL,NULL]::FLOAT4[],
	istdev 			FLOAT4[2] DEFAULT ARRAY[NULL,NULL]::FLOAT4[],
	sparse_init 		INT4[2] DEFAULT ARRAY[NULL,NULL]::INT4[],
	sparse_stdev 		FLOAT4[2] DEFAULT ARRAY[1.,1.]::FLOAT4[],
	init_bias 		FLOAT4[2] DEFAULT ARRAY[0.,0.]::FLOAT4[],
	W_lr_scale 		FLOAT4[2] DEFAULT ARRAY[NULL,NULL]::FLOAT4[],
	b_lr_scale 		FLOAT4[2] DEFAULT ARRAY[NULL,NULL]::FLOAT4[],
	max_col_norm		FLOAT4[2] DEFAULT ARRAY[NULL,NULL]::FLOAT4[],
	weight_decay_coeff	FLOAT4[2] DEFAULT ARRAY[NULL,NULL]::FLOAT4[],
	PRIMARY KEY (layer_id)
) INHERITS (hps3.layer);


CREATE TABLE stochastic.layer_StraightThrough (
	dim			INT4,
	hidden_dim		INT4,
	hidden_activation 	VARCHAR(255) DEFAULT 'tanh',
	expert_activation	VARCHAR(255),
	derive_sigmoid		BOOLEAN DEFAULT TRUE,
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


--DROP TABLE stochastic.layer_SmoothTimesStochastic
CREATE TABLE stochastic.layer_SmoothTimesStochastic (
	dim			INT4,
	hidden_dim		INT4,
	hidden_activation 	VARCHAR(255) DEFAULT 'tanh',
	sparsity_target 	FLOAT4 DEFAULT 0.1,
	sparsity_cost_coeff 	FLOAT4 DEFAULT 0.001,
	noise_beta		FLOAT4 DEFAULT 40.1,
	noise_scale 		FLOAT4,
	noise_stdev		FLOAT4 DEFAULT 7.0,
	noise_normality		FLOAT4 DEFAULT 0.5,
	stochastic_ratio	FLOAT4 DEFAULT 0.5,
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


CREATE TABLE stochastic.layer_NoisyRectifier (
	dim			INT4,
	hidden_dim		INT4,
	hidden_activation 	VARCHAR(255) DEFAULT 'tanh',
	expert_activation	VARCHAR(255) DEFAULT 'linear',
	gater_activation	VARCHAR(255) DEFAULT 'rectifiedlinear',
	sparsity_cost_coeff 	FLOAT4 DEFAULT 1.0,
	sparsity_target		FLOAT4 DEFAULT 0.1,
	sparsity_decay		FLOAT4 DEFAULT 1.0,
	noise_stdev		FLOAT4 DEFAULT 14.0,
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

--DROP TABLE stochastic.layer_BaselineSigmoid
CREATE TABLE stochastic.layer_BaselineSigmoid (
	dim			INT4,
	hidden_dim		INT4,
	hidden_activation 	VARCHAR(255) DEFAULT 'tanh',
	sparsity_target 	FLOAT4 DEFAULT 0.1,
	sparsity_cost_coeff 	FLOAT4 DEFAULT 1.0,
	noise_beta		FLOAT4 DEFAULT 80.1,
	noise_scale 		FLOAT4 DEFAULT 2.1976,
	noise_stdev		FLOAT4 DEFAULT 14.0,
	noise_normality		FLOAT4 DEFAULT 0.5,
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
CREATE TABLE stochastic.cost_conditional1 () INHERITS (hps3.cost_mlp);

CREATE TABLE stochastic.ddm_newsgroups20 (
	which_set	VARCHAR(255), 
	data_path	VARCHAR(255) DEFAULT NULL, 
	valid_ratio	FLOAT4 DEFAULT 0.2,
        sum_to_one	BOOLEAN DEFAULT True,
	one_hot 	BOOLEAN DEFAULT True,
	PRIMARY KEY (ddm_id)
) INHERITS (hps3.ddm);