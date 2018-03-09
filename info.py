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
import logging
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
sns.set_style("dark")
import spynnaker8 as s
import pyNN.utility.plotting as plot

from SpikingConvNet import algorithms, classes, utils
from SpikingConvNet.parameters import *



def info(rc, model):
    rc.rebuild = True
    s.setup(timestep=TIMESTEP)
    scnn = classes.Spinnaker_Network(rc,model)


    scnn.print_parameters()
    s.end()


    """ Display Plots
    """
    try:

        w_1 = scnn.w_layer[1].reshape((-1,model.layers[1].shape[0], model.layers[1].shape[1]))
        utils.plot_heatpmap(rc, w_1, title = "Kernel Weights Layer 1")
    except:
        print("could not plot w_1")
    try:
        w_2 = scnn.w_layer[2].reshape((-1,model.layers[2].shape[0], model.layers[2].shape[1]))
        utils.plot_heatpmap(rc, w_2, title = "Kernel Weights Layer 2")
    except:
        print("could not plot w_2")

    # utils.plot_membran_voltages(v_post_ex, scnn.total_simtime)
    utils.plot_heatpmap(rc, scnn.images, title="Input Patterns")

    # utils.plot_spikes(spiketrains_post_ex)

    plt.show()

""" Test Unit
"""
if __name__ == '__main__':
    print ("Test unit")
