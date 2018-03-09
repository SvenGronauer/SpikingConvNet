"""
    Deep Spiking Convolutional Neural Network
    with STDP Learning Rule on MNIST data
    ______________________________________________________
    Research Internship
    Technical University Munich
    Creator:    Sven Gronauer
    Date:       February 2018

"""
import logging
import argparse
import pickle
import time
import os
import datetime
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import shutil
sns.set_style("dark")

import spynnaker8 as s
import pyNN.utility.plotting as plot

from SpikingConvNet import algorithms, classes, utils
from SpikingConvNet.parameters import *

def training_function(rc, model):
    """ Train spiking neural network """


    if rc.train_svm == True:    # train classifier

        # setup structure and receive spikes from last conv. layer
        s.setup(timestep=TIMESTEP)
        s.set_number_of_neurons_per_core(s.IF_curr_exp, 100)
        rc.rebuild = True
        network = classes.Spinnaker_Network(rc, model)
        s.run(network.total_simtime)
        [spikes_to_classify, voltage] = network.retrieve_data()
        s.end()

        # transform spiketrains to dataset, a datapoint consists of:
        # count number of spikes for each neuron in last layer
        tensor_last_layer = model.tensors[-1]
        utils.plot_spike_activity(rc, spikes_to_classify, tensor_last_layer)
        plt.show()
        X_train = algorithms.spikes_for_classifier(rc,
                                                   tensor_last_layer,
                                                   spikes_to_classify)
        model.classifier.train(X_train, network.y_train)
        # svm.train_SVM(rc, X_train, y_train)

        print "SVM X_train:"
        print X_train
        print("Max(X_train) = {}".format(np.max(X_train)))
        np.savetxt(DATA_PATH+"/SVM_X_train", X_train, delimiter=',')
        network.print_parameters()

    elif rc.train_layer:

        layer = rc.train_layer
        tensor_prev = model.tensors[layer-1]
        tensor = model.tensors[layer]
        neurons_post = tensor[2] / tensor_prev[2]
        stride = model.layers[layer].stride
        kernel_shape = model.layers[layer].shape


        if layer >= 2:
            # Training a deeper layer requires spike times of the previous layer
            # Therefore, the spikes of the previous layer must be determined at first
            s.setup(timestep=TIMESTEP)
            s.set_number_of_neurons_per_core(s.IF_curr_exp, 100)
            rc.rebuild = True
            network = classes.Spinnaker_Network(rc, model)
            s.run(network.total_simtime)
            [spikes_to_window, voltage] = network.retrieve_data()
            s.end()


            # for i, plain in enumerate(spikes_to_window):
            #     print i,plain

            # utils.plot_membrane_voltages(rc, voltage, network.total_simtime)
            utils.plot_spike_activity(rc, spikes_to_window, tensor_prev)
            # plt.show()
            deep_spikes = algorithms.windowed_spikes(spikes_to_window,
                                                    tensor_prev,
                                                    tensor,
                                                    kernel_shape,
                                                    stride)
            # save deep_spikes
            pickle.dump(deep_spikes, open(MODEL_PATH+"deepspikes.p", "wb"))
            rc.logging.info('Calculated Deep Spikes')
            # comment to save processing time
            # utils.plot_spikes(rc, spikes_to_window, title="Spikes of Layer")

        # Now, train the appreciated layer with STDP rules
        s.setup(timestep=TIMESTEP)
        s.set_number_of_neurons_per_core(s.IF_curr_exp, 100)
        rc.rebuild = False

        if layer >=2:
            network = classes.Spinnaker_Network(rc, model,deepspikes=deep_spikes)
        else:
            network = classes.Spinnaker_Network(rc,model)
        network.print_parameters()
        s.run(network.total_simtime)
        try:
            spikes_input, spikes_layer, voltage  = network.retrieve_data()
        except TypeError:
            rc.logging.error("No spikes have been emitted in the last layer")

        # Receive kernel weights and prepare to plot
        w_init = network.w_init.reshape((-1,kernel_shape[0],kernel_shape[1]))
        network.update_kernel_weights()
        w_1 = network.w_layer[layer].reshape((-1,kernel_shape[0],kernel_shape[1]))
        delta = w_1 - w_init

        s.end()
        network.print_parameters()


    """ =============
        Display Plots
        ============= """

    # try:
    #     utils.plot_membrane_voltages(rc, voltage, network.total_simtime)
    # except:
    #     print("not printed voltages !")
    # try:
    #     utils.plot_spikes(rc, spikes_input, title="Spikes of Layer")
    # except:
    #     print("not printed spikes_layer")
    try:
        utils.plot_spikes(rc, spikes_layer, title="Spikes of Layer")
    except:
        print("not printed spikes_layer")

    utils.plot_heatpmap(rc, network.images, title="Input Patterns")


    try:
        utils.plot_heatpmap(rc, delta, title = "Weight Changes", delta=True)
    except:
        print("not printed DELTA weights")

    try:
        utils.plot_heatpmap(rc, w_1, title = "Kernel Weights_1")
    except:
        print("not printed w_1 Kernel weights")

    plt.show()
