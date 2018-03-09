"""
    This module provides classes for creating objects of the neural network
    model and the necessary infracture for building networks on SpiNNaker

    Classes are namely:

    * Layer
    * InputLayer(Layer)
    * ConvLayer(Layer)
    * Classifier(Layer)
    * SpikingModel
    * Spinnaker_Network
"""
import logging
import argparse
import scipy
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random

from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

import spynnaker8 as s
from pyNN.random import RandomDistribution
import pyNN.utility.plotting as plot


from SpikingConvNet.parameters import *
from SpikingConvNet import algorithms, utils

class Layer(object):
    """ Base class for network layers """
    def __init__(self, rc):
        self.rc = rc

class InputLayer(Layer):
    """ Specifies input layer of network """
    def __init__(self, tensor):
        self.tensor = tensor

class ConvLayer(Layer):
    """ Specify a convolutional layer of the network """
    def __init__(self, kernels, shape, stride):
        self.kernels = kernels
        self.shape = shape
        self.stride = stride

class Classifier(Layer):
    """ Holds a linear Support Vector Machine for classifying """
    def __init__(self):
        self.svm = svm.LinearSVC()
        self.trained = False
        try:    # try to restore model from disk
            self.svm = joblib.load(SVM_MODEL_PATH)
            self.trained = True
        except:
            pass

    def train(self, X_train, y_train):
        """ Train parameters of SVM model with given Trainset
        """
        self.svm.fit(X_train, y_train)
        self.trained = True
        y_pred = self.svm.predict(X_train)
        acc = accuracy_score(y_train, y_pred)*100
        print("SVM - Training Accuracy = {}%".format(acc))

        joblib.dump(self.svm, SVM_MODEL_PATH)    # save model to disk

    def predict(self, X_test, y_test):
        """ Determine classification accuracy of SVC with given Testset
        """
        assert self.trained
        y_pred = self.svm.predict(X_test)
        acc = accuracy_score(y_pred, y_test) * 100
        print("Predictions = [{}]".format(y_pred))
        print("SVM - Testing Accuracy = {}%".format(acc))
        cm = confusion_matrix(y_test, y_pred)


        return acc, cm

class SpikingModel(object):
    """ Describes structure of spiking neural network

        A model consists of sequential topology of layers. The first layer is
        provided as an input layer, followed by convolutional layers. The last
        layer should be defined by a Classifier.

        Parameters
        ----------
        input_tensor: tuple of int
            Size of input patterns as 3d-Tensor

        run_control: RunControl object
            Holds information of program flow
    """
    def __init__(self, input_tensor, run_control=None):
        self.input_tensor = input_tensor
        self.tensors = [input_tensor]
        self.rc = run_control
        self.layers = [InputLayer(input_tensor),]   # add input layer by default
        self.number_layers = 0
        self.classifier = None
        self.rc.logging.info('*** Spiking Model built ***')

    def add(self, layer):
        """ Adds sequentially a layer to the network model """

        if isinstance(layer, ConvLayer):
            self.layers.append(layer)
        if isinstance(layer, Classifier):
            self.classifier = layer
        self._calculate_tensors()
        self.number_layers = len(self.layers)-1 # subract input layer

    def _calculate_tensors(self):
        """ Determines automatically tensors for each layer """

        self.tensors = [self.input_tensor]
        t_prev = self.input_tensor
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                d1 = (t_prev[0] - layer.shape[0]) / layer.stride + 1
                d2 = (t_prev[1] - layer.shape[0]) / layer.stride + 1
                d3 = layer.kernels * t_prev[2]
                t_prev = (d1,d2,d3)
                self.tensors.append((d1,d2,d3))

    def print_structure(self):
        """ Print struture of model in console """

        print("Structure of Spiking Conv Net ")
        print("Layers = {}".format(self.number_layers))

        print("Structure: ")
        print("* Input - Tensor={}".format(self.tensors[0]))
        for i, layer in enumerate(self.layers[1:]):
            print("* Layer {} - {} Kernels with shape ({}), stride={}, Tensor={}"\
                .format(i+1,layer.kernels,layer.shape, layer.stride, self.tensors[i+1]))
        if self.classifier:
            print("* Classifier")


class Spinnaker_Network(object):
    """ Class for implementing neural network model on SpiNNaker

        The following steps are processed through calling the class constructor
        (based on PyNN basic setup structure)

        #. Initialize with constructor
        #. Load datasets (Train and Testset) from local files
        #. Load previously calculated weights for layers (if given in /model)
        #. Create populations
        #. Build STDP-model
        #. Build projections between populations
        #. Setup recordings

        The following methods must be called from external fuction(s):

        * update_kernel_weights() - Determine current weights in STDP trained layer
        * retrieve_data() - Receive recorded data from SpiNNaker
        * print_parameters() - Display information of neural network

        Parameters
        ----------
        runcotrol:  RunControl object
            Structure that contains basic information for program flow such as
            passed args from terminal command, backup commands, building options
            for SpiNNaker network

        model: SpikingModel object
            predefined model of spiking neural network

        deepspikes: Spiketrain object
            training a deeper layer requires preprocessed spikes from previous
            layer, hence training of Spiking Neural Network is done layer by
            layer

    """

    def __init__(self, runcontrol, model, deepspikes=None):
        """ Initialize framework for SpiNNaker implementation
        """
        self.model = model
        self.rc = runcontrol
        self.deep_spikes = deepspikes
        self.total_simtime  = 0
        runcontrol.logging.info('Build Deep SCNN')
        runcontrol.logging.info('mode = {}'.format(self.rc.args.mode))

        self._load_data_sets()  # into self.X_train and y_train
        self.w_layer = self._load_kernel_weights()
        self._build_populations()
        self.stdp_model = self._build_stdp_model()
        self.projections = self._build_projections()
        self._recordings()


    def _load_data_sets(self):
        """ Load datasets from files

            datasets built by utils.load_MNIST_digits_shuffled() are locally
            stored in directory data/
        """

        try:
            self.X_train = np.loadtxt(DATA_PATH+"X_train",
                                      dtype = np.float32, delimiter=',')
            self.y_train = np.loadtxt(DATA_PATH+"y_train",
                                      dtype = np.float32, delimiter=',')
            self.rc.size_train_set = self.X_train.shape[0]
        except:
            self.rc.logging.critical("Training data not found")
            raise RuntimeError("Training data not found")
        try:
            self.X_test = np.loadtxt(DATA_PATH+"X_test",
                                      dtype = np.float32, delimiter=',')
            self.y_test = np.loadtxt(DATA_PATH+"y_test",
                                      dtype = np.float32, delimiter=',')
            self.rc.size_test_set = self.X_test.shape[0]
        except:
            self.rc.logging.critical("Testing data not found")
            raise RuntimeError("Testing data not found")

        # store up to 64 of first input patterns for later display plot
        self.images = []
        for i in range(self.X_train.shape[0]):
            self.images.append(self.X_train[i,:].reshape((28,28)))
            if i == 63:
                break
        self.rc.logging.info('Successfully loaded Datasets')


    def _load_kernel_weights(self):
        """ Load previously learned weights from files
            located in directory /model/
        """

        w_layer = [[] for i in range(self.model.number_layers+1)]
        for i in range(self.model.number_layers):
            path = MODEL_PATH+"w_layer_{}".format(i+1)
            try:    # format (neurons, 64)
                w = np.loadtxt(path, dtype = np.float32, delimiter=',')
                w_layer[i+1]  = w
            except:
                self.rc.logging.error('Failed to load weights for Layer {}'.format(i+1))
                break
        return w_layer


    def _build_populations(self):
        """ Build Populations

            Two options are available for the building process:

            #.) Rebuild
                The spiking network is flattend and no time-sclices of spiketrains
                are accomplished
                This Rebuild option is afforded to receive spikes for training a
                deeper layer, to train the SVM classifier or for applying the
                testset to the network

            #.) Train_Layer
                For this instance, only two populations have to be built:
                the first population represents the time-scliced input (that is
                the over time windowed kernel);
                the second population is the respective kernel that are trained
                with STDP Rule
        """


        if self.rc.rebuild == True:

            self.rc.logging.info('Mode = Rebuild')

            if self.rc.train_layer:    # training mode, load train set
                self.total_simtime = self.rc.size_train_set*SIM_INTERVAL
                rebuild_layers = self.rc.train_layer - 1    # number of layers to rebuild

            elif self.rc.train_svm:
                self.total_simtime = self.rc.size_train_set*SIM_INTERVAL
                rebuild_layers = self.model.number_layers

            else:   # load test set
                self.total_simtime = self.rc.size_test_set*SIM_INTERVAL
                rebuild_layers = self.model.number_layers


            self.populations = []

            size_input = reduce(lambda x,y: x*y, self.model.input_tensor)
            spiketrains = algorithms.input_flattend_spikes(self.X_train,
                                                           self.model.input_tensor,
                                                           self.model.layers[1].shape)
            pop = s.Population(size_input,
                                 s.SpikeSourceArray(spike_times=spiketrains),
                                 label="input_neurons")
            self.populations.append(pop)


            for i in xrange(1,rebuild_layers+1): # remember [1,x[ -> x is excluded

                try:
                    CELLS_LAYER = eval("REBUILD_LAYER_{}".format(i))
                    self.rc.logging.info("REBUILD_LAYER_{}".format(i))
                except:
                    raise NotImplementedError("Parameters for Layer {} \
                                        not found in parameters.py".format(i))

                size_layer = reduce(lambda x,y: x*y, self.model.tensors[i])
                # size_layer = reduce(lambda x,y: x*y, TENSOR_LAYER_1)
                self.rc.logging.info("Size Layer {} = {}".format(i,size_layer))
                self.populations.append(  s.Population(size_layer,
                                            s.IF_curr_exp(**CELLS_LAYER),
                                            label="neurons_layer_{}".format(i)))

            if self.rc.args.debug:
                print("DEBUG (_build_populations) - Input Layer Spiketrains ")
                for i, sp in enumerate(spiketrains):
                    print("{} - {}".format(i,sp))


        elif self.rc.train_layer:
            """ Train a layer with STDP Rule

                only two populations have to be built on SpiNNaker:

                #.) Population that generates spikes
                #.) Post-Neurons which kernels are trained with STDP mechanism

                before this code is executed, the spike times for the generated spikes
                must be determined and passed to the network (deepspikes parameter)
            """

            layer = self.rc.train_layer
            tensor_prev = self.model.tensors[layer-1]
            tensor = self.model.tensors[layer]
            neurons_post = tensor[2] / tensor_prev[2]
            stride = self.model.layers[layer].stride
            kernel_shape = self.model.layers[layer].shape

            # calculate number of strides per input pattern
            windows = ((tensor_prev[0]-tensor[0])/stride+1)**2 * tensor_prev[2]

            self.total_simtime = self.rc.size_train_set * SIM_INTERVAL * windows
            self.rc.logging.info('Mode = Train layer {}'.format(layer))

            number_pre_neurons = kernel_shape[0]*kernel_shape[1]
            if layer == 1:
                spiketrains = algorithms.input_windowed_spikes(self.X_train,
                                                               self.model.input_tensor,
                                                               kernel_shape,
                                                               stride)
            else:   # layer > 1, take previously calculated spikes
                spiketrains = self.deep_spikes

            try:
                CELLS = eval("TRAIN_LAYER_{}".format(layer))
            except:
                raise NotImplementedError("Parameters for Layer {} not found in parameters.py".format(i+1))

            self.neurons_input = s.Population(number_pre_neurons,
                                             s.SpikeSourceArray(spike_times=spiketrains),
                                             label="input_neurons")
            self.neurons_layer = s.Population(neurons_post,
                                              s.IF_curr_exp(**CELLS),
                                              label="neurons_train_layer_{}".format(layer))

        else:
            self.rc.logging.critical('failed to build populations')
            raise RuntimeError("failed to build populations ")

        self.rc.logging.info('Successfully built populations')


    def _build_stdp_model(self):
        """ Build STDP Model based on the parameters defined in parameters.py

            Weights for the layer to train are initialized randomly by Gaussian
            Distribution. As timing rule for the spike based learning is used
            the Spike Pair Rule with an additive weight dependence.

            Returns
            -------
            STDPMechanism object
        """

        if self.rc.train_layer:

            layer = self.rc.train_layer
            try:
                timing_rule = s.SpikePairRule(tau_plus=eval("TAU_PLUS_LAYER_{}".format(layer)),

                                              tau_minus=eval("TAU_MINUS_LAYER_{}".format(layer)),
                                              A_plus=eval("A_PLUS_LAYER_{}".format(layer)),
                                              A_minus=eval("A_MINUS_LAYER_{}".format(layer)))
            except:
                raise NotImplementedError("Timing rule for Layer {} not found in parameters.py".format(layer))

            try:
                # MultiplicativeWeightDependence
                # AdditiveWeightDependence
                weight_rule = s.AdditiveWeightDependence    (w_max=eval("W_MAX_STDP_LAYER_{}".format(layer)),
                                                         w_min=eval("W_MIN_STDP_LAYER_{}".format(layer)))
            except:
                raise NotImplementedError("weight rule for Layer {} not found in parameters.py".format(layer))

            neurons = self.model.layers[layer].kernels
            kernel_shape = self.model.layers[layer].shape

            try:
                w_shape = (neurons, kernel_shape[0]*kernel_shape[1])   # (4,64)
                self.w_init = np.random.normal(loc=eval("INIT_WEIGHT_MEAN_LAYER_{}".format(layer)),
                                           scale=eval("SIGMA_LAYER_{}".format(layer)),
                                           size=w_shape)
            except:
                raise NotImplementedError("random. parameters for Layer {} not found in parameters.py".format(layer))

            return s.STDPMechanism(timing_dependence=timing_rule,
                                    weight_dependence=weight_rule,
                                    delay=DELAY_SYNAPSE_CONNECTION)

        else:
            return None

    def _build_projections(self):
        """ Build Projections between populations

            Depending on the mode (rebuild or train) layers are connected by
            sparse connections. The particular connection lists are obtained by
            calling the algorithms module.

            Returns
            -------
            Projections
        """

        if self.rc.rebuild == True: # for training SVM, prepare Layer and testing

            if self.rc.train_layer:    # training mode
                rebuild_layers = self.rc.train_layer - 1    # number of layers to rebuild
            else:   #  test mode
                rebuild_layers = self.model.number_layers

            projections = []
            self.rc.logging.info("Number of projections = {}".format(rebuild_layers))
            # Loop over all Layers to build projections
            for layer in xrange(1, rebuild_layers+1):
                try:
                    conn = algorithms.rebuild_fixed_connections(
                        self.model.tensors[layer-1],    # tensor previous layer
                        self.model.tensors[layer],      # tensor current layer
                        self.model.layers[layer].shape, # kernel shape
                        self.model.layers[layer].stride,# stride
                        self.w_layer[layer])
                except:
                    self.rc.logging.critical("Could not rebuild fixed \
                    connections between Layers {} and {}".format(layer-1,layer))
                    raise RuntimeError("Could not rebuild fixed connections")

                # exhibitory Layer
                pro = s.Projection(self.populations[layer-1],
                                   self.populations[layer],
                                   connector=s.FromListConnector(conn),
                                   receptor_type="excitatory",
                                   label="Layer {} Connections - fixed".format(layer))
                projections.append(pro)

            self.rc.logging.info('Successfully built Projections')
            return projections


        # training - build fresh stdp network
        elif self.rc.train_layer:
            layer = self.rc.train_layer
            tensor_prev = self.model.tensors[layer-1]
            tensor = self.model.tensors[layer]
            kernel = self.model.layers[layer].shape
            number_pre_neurons = kernel[0]*kernel[1]
            number_post_neurons = tensor[2] / tensor_prev[2]

            connect_list = []
            for post_neuron in range(number_post_neurons):
                for pre in range(number_pre_neurons):
                    w = self.w_init[post_neuron, pre]
                    conn = [pre, post_neuron, w, DELAY_SYNAPSE_CONNECTION]
                    connect_list.append(conn)

            # STDP connections
            self._projections_stdp = s.Projection(
                self.neurons_input, self.neurons_layer,
                connector=s.FromListConnector(connect_list),
                synapse_type=self.stdp_model,
                receptor_type="excitatory",
                label="STDP Connections")

            # inhibitory connections
            s.Projection(
                self.neurons_layer, self.neurons_layer,
                connector=s.AllToAllConnector(allow_self_connections=False),
                synapse_type=s.StaticSynapse(weight=WEIGHT_INH_CON, delay=1.0),
                receptor_type="inhibitory",
                label="Lateral inhibition")

        else:
            self.rc.logging.critical('Did not build Projections')
            return



    def _recordings(self):
        """ Record spikes and voltages of populations
        """

        if self.rc.rebuild:
            self.populations[-1].record(variables=["spikes","v"])   # only record last layer

        elif self.rc.train_layer:
            self.neurons_input.record("spikes")
            self.neurons_layer.record(variables=["spikes","v"])

        else:
            self.rc.logging.critical('Recording failed')


    """ ================
        Public Functions
        ================
    """

    def update_kernel_weights(self):
        """ Update the internal variables of weights

            Receive actual weight values of trained STDP layer from SpiNNaker
            board and store to self.w_layer variable

            Returns
            -------
            weights, np.array, shape = [n_kernels, flattend_weights]
        """

        kernel_weights = []
        return_weights = None

        layer = self.rc.train_layer
        tensor_prev = self.model.tensors[layer-1]
        tensor = self.model.tensors[layer]
        neurons_post = tensor[2] / tensor_prev[2]

        if self.rc.train_layer:

            got = self._projections_stdp.get('weight', 'array')
            # reshape
            for kernel in range(neurons_post):
                kern = got[:, kernel]
                kernel_weights.append(kern)

            return_weights = np.array(kernel_weights) # shape e.g. (4,64)
            self.w_layer[layer] = np.array(kernel_weights)

            try:
                output = MODEL_PATH+"w_layer_{}".format(layer)
                number = self.model.layers[layer].kernels
                w = self.w_layer[layer].reshape((number,-1))
                np.savetxt(output, w, delimiter=',')
            except:
                self.rc.logging.error("Could not save weights of Layer {} to model/".format(layer))

        else:
            self.rc.logging.error("Weights not received")

        return return_weights


    def retrieve_data(self):
        """ Transmit observed data of spikes and voltages from SpiNNaker Board
            to host computer

            Returns
            -------
            spiketrains: SpikeTrain object
                Spiketrains from last layer in neural network

            list: [spikes_in, spikes_1, v_1]

                * spikes_in: spiketimes input layer
                * spikes_1: spikes post-neurons
                * v_1: membrane potentials of post-neurons
        """

        if self.rc.rebuild == True:
            data = self.populations[-1].get_data(variables=["spikes", "v"])
            spiketrains = data.segments[0].spiketrains
            v = data.segments[0].filter(name='v')

            return [spiketrains, v]

        elif self.rc.train_layer:
            data_in = self.neurons_input.get_data("spikes")
            data_1 = self.neurons_layer.get_data(variables=["spikes", "v"])
            spikes_in = data_in.segments[0].spiketrains
            spikes_1 = data_1.segments[0].spiketrains
            v_1 = data_1.segments[0].filter(name='v')
            return [spikes_in, spikes_1, v_1]


    def print_parameters(self):
        """ Print parameters of model and simulation to console"""

        print("**********************************")
        print("* Parameters\n*")
        print("* Simulation time:       {}s".format(self.total_simtime/1000.0))
        print("* Simintervall:          {}".format(SIM_INTERVAL))
        print("* Timestep:              {}".format(TIMESTEP))

        print("* ")
        self.model.print_structure()
        print("**********************************\n\n")


""" Test Unit
"""
if __name__ == '__main__':
    print ("Test unit")
