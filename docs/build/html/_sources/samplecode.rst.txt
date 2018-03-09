Sample Code
===========

This section describes how to setup a simple Deep Spiking Convolutional Neural Network.

Simple 1-Layer ConvNet
----------------------


Let's start with training a simple SCNN with one convolutional Layer. By creating firstly the model structure with the following python code:

.. code-block:: python

	model = SpikingModel(input_tensor=(28,28,1), run_control=rc)
	model.add(ConvLayer(4,shape=(5,5), stride=2))
	model.add(Classifier())

In order to build the network structure on SpiNNaker Hardware, you have to execute commands in the terminal:

.. code-block:: console

	$python main.py --mode loaddata
	$python main.py --mode training --layer 1
	$python main.py --mode training --layer svm
	$python main.py --mode testing





Deeper ConvNet
--------------

Theoritically, as many layers as appreciated can be build. Therefore convolutional layers are added to the model are added in sequential manner.


.. code-block:: python

	model = SpikingModel(input_tensor=(28,28,1), run_control=rc)
	model.add(ConvLayer(4,shape=(5,5), stride=2))
	model.add(ConvLayer(4,shape=(5,5), stride=2))
	...
	model.add(ConvLayer(4,shape=(3,3), stride=2))
	model.add(Classifier())

.. code-block:: console

	$python main.py --mode loaddata
	$python main.py --mode training --layer 1
	$python main.py --mode training --layer 2
	...
	$python main.py --mode training --layer n
	$python main.py --mode training --layer svm
	$python main.py --mode testing


.. note::
	The training of the Network is done layer by layer, hence the input spikes of the currently trained layer depend on the previous layer. 
	So a new simulation cycle is started the previously calculated layers are flattend to achieve parallel computation. 
