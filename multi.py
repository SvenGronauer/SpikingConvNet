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

def parallel_populations(rc, model):
    """ Run Training Simulations in parallel
    """

    if rc.train_svm == True:
        """ Train SVM """

        s.setup(timestep=TIMESTEP)
        s.set_number_of_neurons_per_core(s.IF_curr_exp, 100)
        rc.rebuild = True
        network = classes.Spinnaker_Network(rc, model)
        s.run(network.total_simtime)
        spikes_to_classify = network.retrieve_data()
        s.end()


        tensor_last_layer = model.tensors[-1]
        y_train = network.y_train
        X_train = algorithms.spikes_for_classifier(rc,
                                                   tensor_last_layer,
                                                   spikes_to_classify)
        svm.train_SVM(rc, X_train, y_train)

        print "SVM X_train:"
        print X_train
        np.savetxt(DATA_PATH+"/SVM_X_train", X_train, delimiter=',')

        s.end()
        network.print_parameters()


    elif rc.train_layer == 1:
        """ Train Layer 1
        """

        layer = rc.train_layer
        tensor_prev = model.tensors[layer-1]
        tensor = model.tensors[layer]
        neurons_post = tensor[2] / tensor_prev[2]
        stride = model.layers[layer].stride
        kernel_shape = model.layers[layer].shape


        NUMBER_PARALLEL = 4

        s.setup(timestep=TIMESTEP)
        s.set_number_of_neurons_per_core(s.IF_curr_exp, 100)
        rc.rebuild = False

        networks = list()

        for i in range(NUMBER_PARALLEL):
            networks.append(classes.Spinnaker_Network(rc,model))
        # scnn1.print_parameters()
        s.run(networks[0].total_simtime)

        # spikes_input1, spikes_layer1, voltage1  = scnn1.retrieve_data()
        # spikes_input2, spikes_layer2, voltage2  = scnn2.retrieve_data()

        w_init = np.loadtxt("model/w_init", dtype = np.float32, delimiter=',')
        w = list()
        for i in range(NUMBER_PARALLEL):
            networks[i].update_kernel_weights()
            w.append(networks[i].w_layer_1.reshape((-1,KERNEL_LAYER_1[0],KERNEL_LAYER_1[1])))

        for i in range(NUMBER_PARALLEL):
            tools.plot_heatpmap(rc, w[i], title = "Kernel Weights")


        # tools.plot_heatpmap(rc, w[0], title = "Kernel Weights", delta=True)
        # tools.plot_heatpmap(rc, w[1], title = "Kernel Weights", delta=True)
        # tools.plot_heatpmap(rc, w[2], title = "Kernel Weights", delta=True)
        # tools.plot_heatpmap(rc, w[3], title = "Kernel Weights", delta=True)
        # tools.plot_heatpmap(rc, w_final, title = "Kernel Weights")

        plt.show()

        #save as pickle
        pickle.dump(w, open(MODEL_PATH+"w_multi.p", "wb"))

        s.end()


    elif rc.train_layer == 2:
        """ Train Layer 2
        """

        if not rc.args.no_prepare:
            # receive training spikes for deep layer
            s.setup(timestep=TIMESTEP)
            rc.prepare_layer = 2
            scnn = cnn.Deep_Spiking_CNN_model(rc)
            s.run(scnn.total_simtime)
            _, spikes_to_window = scnn.retrieve_data()
            s.end()
            deep_spikes = algorithms.windowed_spikes(spikes_to_window,
                                                    TENSOR_LAYER_1,
                                                    TENSOR_LAYER_2,
                                                    KERNEL_LAYER_2,
                                                    STRIDE_LAYER_2)
            # save deep_spikes
            pickle.dump(deep_spikes, open(MODEL_PATH+"deepspikes.p", "wb"))

        else:   # load prepared deep spikes for layer 2
            try:
                deep_spikes = pickle.load(open(MODEL_PATH+"deepspikes.p", "rb"))
            except:
                rc.logging.critical("Cannot pickle deepspikes.p")

        # apply training on second layer with calc. spikes
        s.setup(timestep=TIMESTEP)
        rc.prepare_layer = 0
        scnn = cnn.Deep_Spiking_CNN_model(rc, deepspikes=deep_spikes)
        s.run(scnn.total_simtime)
        spikes_input, spikes_layer, voltage = scnn.retrieve_data()

        print ("Deep Spikes:")
        print deep_spikes
        w_2 = scnn.update_kernel_weights()
        w_2 = w_2.reshape((-1,KERNEL_LAYER_2[0],KERNEL_LAYER_2[1]))
        delta = scnn.w_layer_2 - scnn.w_init_layer_2
        delta = delta.reshape((-1,KERNEL_LAYER_2[0],KERNEL_LAYER_2[1]))
        s.end()
        scnn.print_parameters()





""" Test Unit
"""
if __name__ == '__main__':
    print ("Test unit")
    w_shape = (NEURONS_LAYER_1, KERNEL_LAYER_1[0]*KERNEL_LAYER_1[1])
    w_random = np.random.normal(loc=INIT_WEIGHT_MEAN_LAYER_1,
                                   scale=SIGMA_LAYER_1,
                                   size=w_shape)
    # print("save w_init Weights")
    # np.savetxt("model/w_init", w_random, delimiter=',')


    rc = tools.RunControl(args=None)
    w_final = pickle.load(open(MODEL_PATH+"w_final.p", "rb"))




    print ("Normalize Weights")
    w_norm = np.ones((NEURONS_LAYER_1,4,4))
    for i in range(NEURONS_LAYER_1):
        w_norm[i,:,:] = w_final[i,:,:] - np.amin(w_final[i,:,:])
        w_norm[i,:,:] = w_norm[i,:,:] * 5.0 / np.amax(w_norm[i,:,:])
    tools.plot_heatpmap(rc, w_norm, title = "Kernel Weights")

    pickle.dump(w_norm, open(MODEL_PATH+"w_norm.p", "wb"))
    plt.show()
