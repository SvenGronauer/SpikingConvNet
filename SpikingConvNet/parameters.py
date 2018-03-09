"""
    This file holds all important constans and parameters for controlling the
    simulation

    .. note::
        Any adjustments applied to this file have an impact to all other files in
        the project!
"""

SIM_INTERVAL                = 100
IDLE_FIRING_TIME	        = 50
TIMESTEP                    = 1.0	# cannot be <1.0 for STDP
RATE_CODE                   = True

# MNIST Dataset
# LOAD_ALL_DIGITS		    = False	# depreciated -> use SUBSET_DIGITS
SUBSET_DIGITS		    = [0,1,7,8]	# [0,3,8,2] Panda=83.25%
TRAIN_EXAMPLES		    = 1
TEST_EXAMPLES	    	    = 20	# per class

# Debugs & Prints
DEBUG_SPIKETRAINS           = False
PRINT_MEMBRANE		        = True	# only in Learning mode supported

# Inhibitory Connections
DELAY_INH_CON		    = 1.0
WEIGHT_INH_CON		    = 300.0	# enables Post-neurons to learn specific representations

# Input
INTENSITY_THRESHOLD	    = 80

# Misc
PACKAGE         	= "SpikingConvNet/"
DATA_PATH 		= "data/"
MODEL_PATH 		= "model/"
BACKUP_PATH 		= "generated_backups/"
SVM_MODEL_PATH 		= MODEL_PATH+'SVM_model_params.sav'
LOGFILE         	= "logfile_scnn.log"

##################################################################################
# STDP

DELAY_SYNAPSE_CONNECTION    = 1.0	# at least 1.0 for spinnaker

##################################################################################
#   Cell Parameters
#
#   Layer 1
#
TRAIN_LAYER_1 = { 'tau_m': 10.0, 		# time-constant of RC-model -- Leakiness
		'v_thresh': -5.0, 	# thresholdat which neuron spikes
		'tau_refrac': 2060.0, 	# refactory peroid
		'tau_syn_I': 5.0,	# current decay- inhibitory connections
		'i_offset': 0.0,
		'v_reset': -65.0,
		'v_rest': -65.0,
		'cm': 0.5, 		# capacitance of LIF neuron
		'tau_syn_E': 5.0}     # current decay - exhibitory	#tau=4.0 spiketime 30ms

# Cell Parameters
REBUILD_LAYER_1 = { 'tau_m': 10.0, 		# time-constant of RC-model -- Leakiness
		'v_thresh': -10.0, 	# thresholdat which neuron spikes
		'tau_refrac': 1.0, 	# refactory peroid
		'tau_syn_I': 5.0,	# current decay- inhibitory connections
		'i_offset': 0.0,
		'v_reset': -65.0,
		'v_rest': -65.0,
		'cm': 0.5, 		# capacitance of LIF neuron
		'tau_syn_E': 5.0}     # current decay - exhibitory	#tau=4.0 spiketime 30ms

W_MAX_STDP_LAYER_1                  = 1.0
W_MIN_STDP_LAYER_1                  = 0.0
TAU_PLUS_LAYER_1                    = 7.0
TAU_MINUS_LAYER_1                   = 7.0
A_PLUS_LAYER_1                      = 0.020	# positive learning rate 0.020
A_MINUS_LAYER_1                     = 0.020	# negative learning rate .0160
INIT_WEIGHT_MEAN_LAYER_1            = 0.30
SIGMA_LAYER_1 		            = 0.04

##################################################################################
#   Cell Parameters
#
#   Layer 2
#
TRAIN_LAYER_2 = { 'tau_m': 10.0, 		# time-constant of RC-model -- Leakiness
		'v_thresh': -40.0, 	# thresholdat which neuron spikes
		'tau_refrac': 660.0, 	# refactory peroid
		'tau_syn_I': 5.0,	# current decay- inhibitory connections
		'i_offset': 0.0,
		'v_reset': -65.0,
		'v_rest': -65.0,
		'cm': 0.3, 		# capacitance of LIF neuron
		'tau_syn_E': 5.0}     # current decay - exhibitory	#tau=4.0 spiketime 30ms

# Cell Parameters
REBUILD_LAYER_2 = { 'tau_m': 10.0, 		# time-constant of RC-model -- Leakiness
		'v_thresh': -40.0, 	# thresholdat which neuron spikes
		'tau_refrac': 3.0, 	# refactory peroid
		'tau_syn_I': 5.0,	# current decay- inhibitory connections
		'i_offset': 0.0,
		'v_reset': -65.0,
		'v_rest': -65.0,
		'cm': 0.3, 		# capacitance of LIF neuron
		'tau_syn_E': 5.0}     # current decay - exhibitory	#tau=4.0 spiketime 30ms

# Layer 2
W_MAX_STDP_LAYER_2                  = 1.0
W_MIN_STDP_LAYER_2                  = 0.0
TAU_PLUS_LAYER_2                    = 7.0
TAU_MINUS_LAYER_2                   = 7.0
A_PLUS_LAYER_2                      = 0.0200	# positive learning rate
A_MINUS_LAYER_2                     = 0.0150	# negative learning rate
INIT_WEIGHT_MEAN_LAYER_2            = 0.50
SIGMA_LAYER_2 		            = 0.05


##################################################################################
#   Cell Parameters
#
#   Layer 3
#
TRAIN_LAYER_3 = { 'tau_m': 5.0, 		# time-constant of RC-model -- Leakiness
		'v_thresh': -35.0, 	# thresholdat which neuron spikes
		'tau_refrac': 200.0, 	# refactory peroid
		'tau_syn_I': 5.0,	# current decay- inhibitory connections
		'i_offset': 0.0,
		'v_reset': -65.0,
		'v_rest': -65.0,
		'cm': 0.3, 		# capacitance of LIF neuron
		'tau_syn_E': 5.0}     # current decay - exhibitory	#tau=4.0 spiketime 30ms

# Cell Parameters
REBUILD_LAYER_3 = { 'tau_m': 5.0, 		# time-constant of RC-model -- Leakiness
		'v_thresh': -50.0, 	# thresholdat which neuron spikes
		'tau_refrac': 5.0, 	# refactory peroid
		'tau_syn_I': 5.0,	# current decay- inhibitory connections
		'i_offset': 0.0,
		'v_reset': -65.0,
		'v_rest': -65.0,
		'cm': 0.5, 		# capacitance of LIF neuron
		'tau_syn_E': 5.0}     # current decay - exhibitory	#tau=4.0 spiketime 30ms

W_MAX_STDP_LAYER_3                  = 2.5
W_MIN_STDP_LAYER_3                  = 0.0
TAU_PLUS_LAYER_3                    = 7.0
TAU_MINUS_LAYER_3                   = 7.0
A_PLUS_LAYER_3                      = 0.0600	# positive learning rate
A_MINUS_LAYER_3                     = 0.0900	# negative learning rate
INIT_WEIGHT_MEAN_LAYER_3            = 0.50
SIGMA_LAYER_3 		            = 0.05
