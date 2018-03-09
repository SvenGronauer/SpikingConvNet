""" Utilities for controling program flow, data handling and data plotting
"""

import time
import os
import datetime
import logging
import argparse
import itertools
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyNN.utility.plotting as plot
import spynnaker8 as s
import math
import random
import seaborn as sns
sns.set_style("dark")
import mnist

from SpikingConvNet.parameters import *
# from SpikingConvNet import algorithms, classes

import sklearn.metrics
from scipy.ndimage import gaussian_filter


class RunControl(object):
    """ Object for controlling program flow, contains args from console and sets
        up the logging utility

        Parameters
        ----------
        args: ArgumentParser object
            Passed arguments from terminal command

        trainlayer: int, optional
            If not zero, specifies which layer of network to train

        trainsvm: bool, optional
            If given, classifier is trained

        rebuild: bool, optional
            Controls the behaviour of the follow up build of neural network
            (as a variable of programs state machine)

            * ``rebuild==True`` in order to train layer n, the spikes of layer n-1 must be determined
            * ``rebuild==False`` a layer or the Classifier is trained

    """
    def __init__(self, args, trainlayer=0, trainsvm=False, rebuild=False):
        self.args           = args
        self.rebuild        = rebuild
        self.train_layer    = trainlayer
        self.train_svm      = trainsvm
        self.backup         = not args.no_backup and not args.mode == 'info'
        self.timestring     = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        self.backup_path    = BACKUP_PATH + self.timestring +"/"
        self.logging        = self.setup_logger()
        self.size_train_set = 0
        self.size_test_set  = 0

    def setup_logger(self):
        """ Setup logger for tracking infos and errors"""

        name = "SCNN_logging"
        log_file = LOGFILE
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        handler = logging.FileHandler(log_file, mode='w'    )
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        return logger

""" ================
    Image Processing
    ================
"""

def dog_filter(image):
    """ Apply Difference of Gaussian Filter to image

        Parameters
        ----------
        image: np.array, shape=[height, width]
            Image to be transformed

        Returns
        -------
        norm_dog: np.array, shape=[height, width]
            Transformed image
    """
    sigma_1 = 1.1
    sigma_2 = 1.0
    dog = (gaussian_filter(image, sigma=sigma_1) - gaussian_filter(image, sigma=sigma_2)) * 100
    super_threshold_indices = dog < 0
    dog[super_threshold_indices] = 0
    norm_dog = dog / np.amax(dog) * 255

    return norm_dog.astype(int)

""" ==================
    Plotting Functions
    ==================
"""
def plot_spikes(rc, pre, post=None, title="Spikes Plot", path=None):
    """ Plot spikes of given layers

        Parameters
        ----------
        rc: RunControl object
            contains information of backup behaviour

        pre: SpikeTrain object
            Spiketrains of first layer to plot

        Returns
        -------
        norm_dog: np.array, shape=[height, width]
            Transformed image
    """
    f = plt.figure()
    # ax = plt.subplot(121)
    for neuron,spiketrains in enumerate(pre):
        for sp in spiketrains:
            # one spiketrain for each simulation, sim.reset() increments the number
            y = np.ones_like(sp) * neuron   # adjust y-axes
            plt.plot(sp, y, '.', color='b')
    if post != None:
        for neuron,spiketrains in enumerate(post):
            for sp in spiketrains:
                # one spiketrain for each simulation, sim.reset() increments the number
                y = np.ones_like(sp) * neuron   # adjust y-axes
                plt.plot(sp, y, '.', color='r')
    plt.ylabel("Neurons")
    plt.title("Firing of Neurons (Pre=blue, Post=red)")

    # save plot
    if rc.backup:
        f.savefig(rc.backup_path+title+'.svg', dpi=f.dpi)


def plot_heatpmap(rc, list_of_elements, title="Default Title", delta=False):
    """ Plot a matrix of heatmaps

        Parameters
        ----------
        rc: RunControl object
            contains information of backup behaviour

        list_of_elements: list
            List of np.arrays with shape=[n,n]

        title: str
            defines title of figure and name of saved figure on disk

        delta: bool
            if defined, use dirrential color scheme for heatmap plot

        Returns
        -------
        f: figure object
            Object of heatmap
    """
    if delta:   # if defined, plot with new color sheme
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        center = 0
    else:
        # cmap = sns.diverging_palette("Reds")
        center = (W_MAX_STDP_LAYER_1 + W_MIN_STDP_LAYER_1) / 2.0
        cmap = sns.cubehelix_palette(16, start=2, rot=0, dark=0, light=.95, reverse=False)

    dim = int(math.ceil(math.sqrt(len(list_of_elements))))
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    f = plt.figure()
    plt.title(title)
    for i, kernel in enumerate(list_of_elements):
        plt.subplot(dim, dim, i+1)
        sns.heatmap(kernel, cmap=cmap, center=center)
        if i == dim*dim:
            break
        if i == 48:
            break
    # save plot
    if rc.backup:
        f.savefig(rc.backup_path+title+'.svg', dpi=f.dpi)

    return f

def plot_spike_activity(rc, spiketrains, tensor, title="plot_spike_activity"):
    """ Plot the activity of each neurons in the given layer in the first
        simulation interval

        The quantity of spikes for each neuron in the layer (given with tensor
        and its spiketrains) are determined for the first simulation interval
        and then plotted as heatmap

        Parameters
        ----------
        rc: RunControl object
            contains information of backup behaviour

        spiketrains: SpikeTrain Object
            List of np.arrays with shape=[n,n]

        tensor: tuple of int
            Tensor of Layer

        title: str
            defines title of figure and name of saved figure on disk

        Returns
        -------
        f: figure object
            Object of figure
    """
    size_tensor = reduce(lambda x,y: x*y, tensor)
    plain_spiketimes = [[] for i in range(size_tensor)]
    for i,spiketrain in enumerate(spiketrains):
        for spiketime in spiketrain.magnitude:
            plain_spiketimes[i].append(spiketime)

    t_start = 0
    t_end = t_start+SIM_INTERVAL
    counted_spikes = []
    for i in range(size_tensor):
        array = np.array(plain_spiketimes[i]) # list to array

        t1 = array >= t_start   # filter appreciated
        t2 = array < t_end      # time interval

        numer_spikes = len(np.where(np.logical_and(t1, t2))[0])
        counted_spikes.append(numer_spikes)


    a = np.ones((tensor[2],tensor[0],tensor[1]))

    for i in range(size_tensor):
        a1 = i / (tensor[0]*tensor[1])
        a2 = (i % (tensor[0]*tensor[1]))/ tensor[0]
        a3 = i % tensor[0]
        a[a1,a2,a3] = counted_spikes[i]

    if rc.args.debug:
        print("plot_spike_activity - Matrix Counted Spikes")
        print a

    # cmap = sns.diverging_palette("Reds")
    center = np.max(a) / 2.0
    cmap = sns.cubehelix_palette(16, start=2, rot=0, dark=0, light=.95, reverse=False)

    dim = int(math.ceil(math.sqrt(tensor[2])))
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    f = plt.figure()
    plt.title(title)
    for i in range(tensor[2]):
        kernel = a[i,:,:]
        plt.subplot(dim, dim, i+1)
        sns.heatmap(kernel, cmap=cmap, center=center)
        if i == dim*dim:
            break
        if i == 99: # break at 100 representations
            break
    # save plot
    if rc.backup:
        f.savefig(rc.backup_path+title+str(tensor[2])+'.svg', dpi=f.dpi)

    return f

def plot_membrane_voltages(rc, v_data, simtime, title="Membrane potentials"):
    """ Plot the membrane potential

        Parameters
        ----------
        rc: RunControl object
            contains information of backup behaviour

        v_data: Voltage Object
            Contains values of the membrane potentials

        simtime: int
            Maximum simulation time on the x-axis

        title: str
            defines title of figure and name of saved figure on disk

        Returns
        -------
        f: figure object
            Object of figure
    """
    line_properties = [{'color': 'red', 'markersize': 5},
                       {'color': 'blue', 'markersize': 5}]

    f = plot.Figure(
        # plot voltage for first ([0]) neuron
        plot.Panel(v_data[0], ylabel="Membrane potential (mV)",
        yticks=True, xlim=(0, simtime)),

        title="Membrane potentials (mV) - Post-Neurons",
        annotations="Sven Gronauer - Simulated with {}".format(s.name())
    )


def plot_confusion_matrix(rc,
                          cm,
                          normalize=True,
                          title='Confusion_matrix',
                          cmap=plt.cm.Blues):
    """ This function prints and plots a confusion matrix.
        Normalization can be applied by setting ``normalize=True``.

        Parameters
        ----------
        rc: RunControl object
            contains information of backup behaviour

        cm: np.array,
            Contains values of the confusion matrix

        normalize: bool
            If ``True`` Matrix is normalized

        title: str
            defines title of figure and name of saved figure on disk

        cmap: cm object
            Color scheme of the plot

        Returns
        -------
        f: figure object
            Object of figure
    """
    f = plt.figure()
    # cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(SUBSET_DIGITS))
    plt.xticks(tick_marks, SUBSET_DIGITS, rotation=45)
    plt.yticks(tick_marks, SUBSET_DIGITS)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # save plot

    if rc.backup:
        try:
            f.savefig(rc.backup_path+title+'.svg', dpi=f.dpi)
        except:
            rc.logging.error("Could not backup confusion matrix ")
    return f


""" =======================
    Data Handling Functions
    =======================

"""

def load_MNIST_digits_shuffled(rc, mode):
    """ Load MNIST dataset

        load defined number of examples (see parameters.py)
        loaded subset of digits is defined in SUBSET_DIGITS
        shuffle data and return as 2d-arrays

        Parameters
        ----------
        rc: RunControl object
            contains information of backup behaviour

        mode: str

            * ``train_examples`` Examples for Trainset are loaded
            * ``test_examples`` Examples for Testset are loaded

        Returns
        -------
        f: figure object
            Object of figure
    """
    mndata = mnist.MNIST('./python-mnist/data')
    mndata.load_training()

    labels = np.array(mndata.train_labels[:])

    if mode == "train_examples":
        number_examples = TRAIN_EXAMPLES
        rc.logging.info("train_examples - Loading {} examples per class".format(number_examples))
    elif mode == "test_examples":
        number_examples = TEST_EXAMPLES
        rc.logging.info("test_examples - Loading {} examples per class".format(number_examples))


    print("Subset of digits: {}".format(SUBSET_DIGITS))

    M = len(SUBSET_DIGITS) * number_examples
    X_images = np.ones((M,28*28))
    y_train = np.ones((M,1))

    for i, digit in enumerate(SUBSET_DIGITS):
        # receive positions of digits
        y = np.argwhere(labels == digit)[:][:, 0]
        # select random digit from all digits
        y = np.random.choice(y, size=number_examples)

        for j,tar in enumerate(y):
            X_images[i*number_examples+j, :] = np.asarray(mndata.train_images[tar])
            y_train[i*number_examples+j, 0] = digit

    from sklearn.utils import shuffle
    X_images, y_train = shuffle(X_images, y_train)

    if mode == "train_examples":
        np.savetxt(DATA_PATH+"X_train", X_images, delimiter=',')
        np.savetxt(DATA_PATH+"y_train", y_train.ravel(), delimiter=',')

    elif mode == "test_examples":
        np.savetxt(DATA_PATH+"X_test", X_images, delimiter=',')
        np.savetxt(DATA_PATH+"y_test", y_train.ravel(), delimiter=',')

    return [X_images, y_train.ravel()]   # X_images = (N,784); y_train = (N,)



""" =========================
    Code Conversion Functions
    =========================
"""


def convert_time_code(intensity):
    """ Time code spikes - Assign pixel intensity to deterministic time interval

        Parameters
        ----------
        intensity: int, [0,255]
            Pixel intensity

        Returns
        -------
        spiketime: int
            Corresponding spike time for pixel intensity
    """
    return int((1 - intensity / 255.0) * (SIM_INTERVAL-1-IDLE_FIRING_TIME))


def convert_rate_code(intensity, total_intensity=None):
    """ Rate code spikes - Assign pixel intensity to stochastic spike times

        calculate spike times dependend on pixel intensity of pre-neuron

        Parameters
        ----------
        intensity: int, [0,255]
            Pixel intensity

        total_intensity: int
            Sum of intensities of input pattern for normalization

        Returns
        -------
        spiketimes: list, of int
            Corresponding spike times for pixel intensity
    """
    spiketimes = []
    fire_factor = 10

    if total_intensity != None:
        fire_factor = int(6.0 * total_intensity / 2500.0)
        # 2000.0 for DoG, 2500 for unprocessed input
        fire_factor = 20 if fire_factor > 20 else fire_factor

    if intensity >= INTENSITY_THRESHOLD:

        t_d = int(255.0/intensity * fire_factor)
        if t_d >= (SIM_INTERVAL-IDLE_FIRING_TIME):
            spiketimes.append(random.randint(0,SIM_INTERVAL-IDLE_FIRING_TIME))
            return spiketimes

        offset = random.randint(0,t_d)
        for i in range(10):
            z = i * t_d + offset
            if z >= (SIM_INTERVAL-IDLE_FIRING_TIME):
                break
            else:
                spiketimes.append(z)

    else:   # if intensity < INTENSITY_THRESHOLD:
        # add  single stochastic spike for intensities below threshold
        spiketimes.append(random.randint(0,SIM_INTERVAL-IDLE_FIRING_TIME))

    return spiketimes



""" Test Unit
"""
if __name__ == '__main__':
    print ("Test unit")

    print("test convert_time_code")
    print convert_time_code_pol(255)
    print convert_time_code_pol(240)
    print convert_time_code_pol(230)

    print ("rate code:")
    print convert_rate_code(255)
    print convert_rate_code(251)
    print convert_rate_code(240)
    print convert_rate_code(230)
    print convert_rate_code(180)
    print convert_rate_code(160)
    print convert_rate_code(80)
    print convert_rate_code(20)


    # print("test")
    # a = np.array([1,2,3,4,5,6,7,8,9,10])
    # print np.random.choice(a, size=3)


    # print("test confusion_matrix")
    # y_test = np.array([0,0,2,3,8])
    # y_pred = np.array([0,1,2,3,8])
    # cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(cm)
    # plt.show()
