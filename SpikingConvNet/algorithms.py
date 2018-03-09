""" Algorithms are provided for

    * Generating sparse connections between populations
    * Slice a layer into windows and split windows over time
"""
import numpy as np
import spynnaker8 as s
import math
import random
import sklearn.metrics
from SpikingConvNet import utils
from SpikingConvNet.parameters import *

def windowed_spikes(spiketrains_input,
                    tensor_first,
                    tensor_second,
                    kernel_shape,
                    stride):
    """ Create windowed SpikeSourceArray
        for instance in training deeper layers of the network

        Input spiketrains are windowed over time, post-neurons are presented only
        a subset of the input spiketrains. The corresponding output spiketimes
        depend on calculated spikes (spiketrains_input) of previous layer.

        Parameters
        ----------
        spiketrains_input: SpikeTrain object
            Spiketrains from previous layer

        tensor_first: tuple of int
            Dimensions of previous layer

        tensor_second: tuple of int
            Dimensions of posterior layer

        kernel_shape: tuple of int
            Kernel shape

        stride: int
            Specified stride over convolved layer

        Returns
        ----------
        spiketrains: np.array, shape = [n_examples*windows, spike_times]
            Each datapoint contains precise spike times for each neuron
    """
    size_first = reduce(lambda x,y: x*y, tensor_first)
    size_second = reduce(lambda x,y: x*y, tensor_second)
    kernel_size = kernel_shape[0] * kernel_shape[1]

    # convert neo spiketrains to plain lists
    plain_spiketimes = [[] for i in range(size_first)]
    for i,spiketrain in enumerate(spiketrains_input):
        for spiketime in spiketrain.magnitude:
            plain_spiketimes[i].append(spiketime)

    # print "Plain Spike times:"
    # for i, plain in enumerate(spiketrains_input):
    #     print plain

    #calculate step offsets over input tensor
    y1_offsets = []
    for y in range(tensor_first[0]-kernel_shape[0]+1): # 5-3+1=3
        if y % stride == 0:
            y1_offsets.append(y)
    # print "y1_offsets", y1_offsets
    x1_offsets = []
    for x in range(tensor_first[1]-kernel_shape[1]+1): # 5-3+1=3
        if x % stride == 0:
            x1_offsets.append(x)
    # print "x1_offsets", x1_offsets

    spiketrains = [[] for i in range(kernel_size)]  # 3*3
    t_start = 0
    t_end = t_start+SIM_INTERVAL
    clock = 0
    number_examples = TRAIN_EXAMPLES * len(SUBSET_DIGITS)

    for step in range(number_examples):                 # iterate over examples
        for depth in range(tensor_first[2]):            # iterate over depth

            for offset_y in y1_offsets:                 # iterate over 2d-layers
                for offset_x in x1_offsets:

                    for y in range(kernel_shape[0]):    # iterate over window
                        for x in range(kernel_shape[1]):

                            # determine actual postion at previous layer
                            pos = (y+offset_y)*tensor_first[0] + x+offset_x
                            # get all spike times for specific postion
                            array = np.array(plain_spiketimes[pos])
                            # consider only appreciated time interval
                            t1 = array >= t_start   # filter spike times
                            t2 = array < t_end      # filter spike times

                            positions = np.where(np.logical_and(t1, t2))[0]
                            spikes = array[positions]   # receive spikes for intervall

                            # re-calculate spike time for time-splitted windows
                            spikes %= SIM_INTERVAL
                            spikes += clock

                            for spike in spikes:
                                spiketrains[y*kernel_shape[0]+x].append(spike)

                    clock += SIM_INTERVAL
        t_start += SIM_INTERVAL
        t_end += SIM_INTERVAL
    return spiketrains

def rebuild_inhibitory_connections(tensor_prev,
                                  tensor_layer,
                                  inhib_weight):
    """ Construct inhibitory connections within flattened layers

        Parameters
        ----------
        tensor_prev: tuple of int
            Dimensions of previous layer

        tensor_layer: tuple of int
            Dimensions of actual layer

        inhib_weight: float32
            fixed weight for inhibitory connection

        Returns
        ----------
        inhib_connection_list: list of [position_1, position_2, weight , delay]
            list for s.FromListConnector()
    """

    layer_offset = tensor_layer[0] * tensor_layer[1]
    kernel_multiplier = tensor_prev[2]
    number_of_kernels = tensor_layer[2] / tensor_prev[2]

    size_layer = tensor_layer[0]*tensor_layer[1]*tensor_layer[2]
    inhib_connection_list = []

    for position in range(size_layer):
        pos_in_grid = position % layer_offset

        for k in range(number_of_kernels):
            pos_to_connect = pos_in_grid + k * layer_offset * kernel_multiplier

            if position != pos_to_connect:
                conn = [position, pos_to_connect, inhib_weight, DELAY_SYNAPSE_CONNECTION]
                inhib_connection_list.append(conn)
    return inhib_connection_list



def rebuild_fixed_connections(tensor_first,
                              tensor_second,
                              kernel_shape,
                              stride,
                              weights_tensor):
    """ Construct fixed connections between flattened layers

        Take previously learned STDP weights and establish fixed weights.
        The computation is handled now in parallel

        Parameters
        ----------
        tensor_first: tuple of int
            Dimensions of previous layer

        tensor_second: tuple of int
            Dimensions of posterior layer

        kernel_shape: tuple of int
            Kernel shape

        stride: int
            Specified stride over convolved layer

        weights_tensor: np.array, shape=[n_kernel, kernel_height*kernel_width]
            Previously trained STDP weights, now initialized as fixed weights

        Returns
        ----------
        connection_list: list of [position_1, position_2, weight , delay]
            list for s.FromListConnector()
    """
    size_first = reduce(lambda x,y: x*y, tensor_first)
    size_second = reduce(lambda x,y: x*y, tensor_second)

    number_kernels = tensor_second[2] / tensor_first[2]
    depth_per_kernel = tensor_first[2]

    connection_list = []

    #calculate step offsets
    y1_offsets = []
    for y in range(tensor_first[0]-kernel_shape[0]+1): # 28-8=20
        if y % stride == 0:
            y1_offsets.append(y)
    x1_offsets = []
    for x in range(tensor_first[1]-kernel_shape[1]+1): # 28-8=20
        if x % stride == 0:
            x1_offsets.append(x)

    for kernel in range(number_kernels):    # iterate over all kernels
        # kernel convolves over several 2d-layers
        for d_offset in range(depth_per_kernel):

            depth = depth_per_kernel * kernel + d_offset
            # iterate over windows of 2d-layer
            for y2, y1_offset in enumerate(y1_offsets):
                for x2, x1_offset in enumerate(x1_offsets):

                    pos2 =  y2 * tensor_second[0] + x2 \
                            + tensor_second[0] * tensor_second[1] * depth

                    # iterate over single window of shape=[kernel_heigt, kernel_width]
                    for y1 in range(kernel_shape[0]):
                        for x1 in range(kernel_shape[1]):
                            pos1 = (y1+y1_offset)*tensor_first[0] + (x1+x1_offset)

                            w = weights_tensor[kernel, y1*kernel_shape[0]+x1]
                            conn = [pos1, pos2, w, DELAY_SYNAPSE_CONNECTION]
                            connection_list.append(conn)
                            # print conn
    return connection_list


def input_windowed_spikes(X_train, tensor_input, kernel_shape, stride):
    """ Create windowed SpikeSourceArray for input neurons
        for instance in training layers of the network

        Input patterns are windowed over time, post-neurons are presented only
        a subset of the input pattern. The corresponding spiketimes depend on
        pixel intensities and are stochastically rate-coded.


        Parameters
        ----------
        X_train: np.array, shape = [n_examples, image_intensities.flatten()]

        Returns
        ----------
        spiketrains: np.array, shape = [n_examples*n_windows, spike_times]
            Each datapoint contains precise spike times for each input neuron
    """

    # h = input_tensor[0]
    # w = input_tensor[1]

    #calculate step offsets
    offsets_vertical = []
    for y in range(tensor_input[0]-kernel_shape[0]+1): # 28-8=20
        if y % stride == 0:
            offsets_vertical.append(y)
    offsets_horizontal = []
    for x in range(tensor_input[1]-kernel_shape[1]+1): # 28-8=20
        if x % stride == 0:
            offsets_horizontal.append(x)

    print("offsets_horizontal = {}".format(offsets_horizontal))
    print("offsets_vertical = {}".format(offsets_vertical))

    kernel_size = kernel_shape[0] * kernel_shape[1]
    spiketrains = [[] for i in range(kernel_size)]
    time_multiplier = 0
    tots = []
    for n, datapoint in enumerate(X_train):

        image = datapoint.reshape((28,28))
        # CHANGED
        # image = utils.dog_filter(data.reshape((28,28)))
        # iterate over all WINDOWS
        for offset_y in offsets_vertical:
            for offset_x in offsets_horizontal:

                # calculate sum of intensities over considered window
                total_int = np.sum(image[offset_y : (offset_y + kernel_shape[0]),
                                    offset_x : (offset_x + kernel_shape[1])])
                tots.append(total_int)

                # iterate over Kernel; get Pixel intensities
                for y in range(kernel_shape[0]):
                    for x in range(kernel_shape[1]):
                        intensity = image[y + offset_y, x + offset_x]

                        if RATE_CODE:   # rate code spikes - stochastic
                            spiketimes = utils.convert_rate_code(intensity, total_int)
                            for spiketime in spiketimes:
                                spiketrains[y*kernel_shape[1]+x].append( \
                                spiketime+ time_multiplier * SIM_INTERVAL)

                        else:       # time code spikes - deterministic
                            if intensity >= INTENSITY_THRESHOLD:
                                spike_time = utils.convert_time_code(intensity) \
                                             + time_multiplier * SIM_INTERVAL
                                spiketrains[y*kernel_shape[1]+x].append(spike_time)
                time_multiplier += 1    # track time intervals for spiketrains
#
    print("ave total intensities: ", np.mean(tots))
    print("max total intensities: ", np.max(tots))

    return spiketrains

def input_flattend_spikes(X_train, tensor_input, kernel_shape):
    """ Create flattened SpikeSourceArray for input neurons
        for instance in rebuilding network

        For rebuilding the network structure the input neurons are not windowed
        over time. Instead, the input layer is flattened and the entire image is
        presented to the network in each simulation interval.
        The corresponding spiketimes depend on pixel intensities and are
        stochastically rate-coded.

        Parameters
        ----------
        X_train: np.array, shape = [n_examples, image_intensities.flatten()]
            dataset of MNIST input images as 2d-array

        tensor_input: tuple of int
            Dimensions of input layer

        kernel_shape: tuple of int
            Kernel shape

        Returns
        -------
        spiketrains: np.array, shape = [n_examples, spike_times]
            Each datapoint contains precise spike times for each input neuron
    """
    number_input_neurons = reduce(lambda x,y: x*y, tensor_input)
    spiketrains = [[] for i in range(number_input_neurons)]
    time_multiplier = 0

    for n, datapoint in enumerate(X_train):

        image = datapoint.reshape((28,28))
        # CHANGED apply dog filter
        # image = utils.dog_filter(datapoint.reshape((28,28)))

        # iterate over Kernel; get Pixel intensities
        for y in range(tensor_input[0]):
            for x in range(tensor_input[1]):
                intensity = image[y, x]

                if RATE_CODE:   # rate code spikes - stochastic
                    spiketimes = utils.convert_rate_code(intensity)
                    for spiketime in spiketimes:
                        spiketrains[y*tensor_input[1]+x].append( \
                        spiketime+ time_multiplier * SIM_INTERVAL)

                else:       # time code spikes - deterministic
                    if intensity >= INTENSITY_THRESHOLD:
                        spike_time = utils.convert_time_code(intensity) \
                                     + time_multiplier * SIM_INTERVAL
                        self.spiketrains[y*KERNEL_SHAPE[1]+x].append(spike_time)
        time_multiplier += 1    # track time intervals for spiketrains

    return spiketrains


def spikes_for_classifier(rc, tensor, spiketrains):
    """ Transform spiketrains to plain two-dimensional dataset

        to reduce the power of Support Vector Machine, the quantity of spikes
        within each simulation interval is counted for each neuron in the last
        layer

        Parameters
        ----------
        tensor: tuple of int
            Tensor of last Convolutional layer in network

        spiketrains: SpikeTrain object
            Retrieved spikes from last layer on SpiNNaker board
            SpikeTrains objects are extracted from Neo Block Segments

        Returns
        ----------
        X: np.array, shape = [datapoints, n_neurons_last_layer]
            Each datapoint contains number of spikes for each post-neuron
            within one sim interval
    """

    if rc.args.mode == "training":
        number_inputs = rc.size_train_set
        print "training"
        print "number_inputs =", number_inputs
    else:
        number_inputs = rc.size_test_set
        print "number_inputs =", number_inputs

    t_start = 0
    t_end = t_start+SIM_INTERVAL
    data = []   # stores spikes for each simulation intervall

    for inp in range(number_inputs):
        values = [] # store number of spikes for each neuron
        t_end = t_start+SIM_INTERVAL    # consider only spikes within interval

        for n,spiketrain in enumerate(spiketrains):

            t1 = spiketrain >= t_start      # get positions lower bound
            t2 = spiketrain < t_end         # get positions upper bound
            # count spikes in interval
            number_spikes = len(np.where(np.logical_and(t1, t2))[0])
            values.append(number_spikes)

        data.append(values)
        t_start += SIM_INTERVAL
        t_end = t_start + SIM_INTERVAL

    X = np.array(data)
    if rc.args.debug:
        print("X=")
        print X
        print ("X.Shape={}".format(X.shape))
    rc.logging.info("Spiketrains for Classifier: X.shape={}".format(X.shape))

    return X


""" Test Unit
"""
if __name__ == '__main__':
    print ("Test unit")

    # windowed_spikes(spiketrains_input,
    #                     tensor_first,
    #                     tensor_second,
    #                     kernel_shape,
    #                     stride):
