"""
    Deep Spiking Convolutional Neural Network
    with STDP Learning Rule on MNIST data
    ______________________________________________________
    Research Internship
    Technical University Munich
    Creator:    Sven Gronauer
    Date:       February 2018

"""
import argparse
# import logging
import numpy as np
# import scipy
import matplotlib.pyplot as plt
import pylab
# import seaborn as sns
# sns.set_style("dark")
import spynnaker8 as s
import pyNN.utility.plotting as plot
import pickle
import sklearn


from SpikingConvNet import algorithms, classes, utils
from SpikingConvNet.parameters import *



def testing_function(rc, model):
    """ Apply testset on trained network """

    s.setup(timestep=TIMESTEP)
    network = classes.Spinnaker_Network(rc, model)

    network.print_parameters()
    s.run(network.total_simtime)
    spikes_layer, voltage = network.retrieve_data()

    try:    # plot spike activity of layer as heatmap
        utils.plot_spike_activity(rc,spikes_layer, model.tensors[-1])
        pickle.dump(spikes_layer, open(MODEL_PATH+"spikes_layer_1.p", "wb"))
    except:
        rc.logging.error("Capturing Spike Activity Failed")
    s.end()

    network.print_parameters()


    """ SVM
    """

    X_test = algorithms.spikes_for_classifier(rc,
                                               model.tensors[-1],
                                               spikes_layer)
    print("X_test")
    print X_test
    print("Maximum X_test")
    print np.max(X_test)


    score, confusion_matrix = model.classifier.predict(X_test, network.y_test)
    try:
        utils.plot_confusion_matrix(rc, confusion_matrix)
    except:
        print("could not print confusion matrix")

    """ Display Plots
    """
    # utils.plot_membrane_voltages(v_post_ex, network.total_simtime)
    # utils.plot_spikes(rc, spikes_layer_1, title="Spikes of Layer 1")
    # utils.plot_spikes(rc, spikes_layer_2, title="Spikes of Layer 2")
    utils.plot_heatpmap(rc, network.images, title="Input Patterns")

    plt.show()
