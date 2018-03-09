Miscellaneous
=============

Build documentation
-------------------

How to build the documentation:

#. Change directory::

	$ cd docs/

#. If major changes have been submitted to the code, then render::

	$ sphinx-apidoc -f -o source/ ../SpikingConvNet/

#. Execute to create HTML Documentation in ``/build/html`` directory::

	$ make html

#. Excute to create PDF of Documentation in ``/build/latex`` directory::

	$ make latexpdf

Obtain MNIST-Dataset
--------------------

thanks to *https://pypi.python.org/pypi/python-mnist*

Get the package from PyPi::

        $ pip install python-mnist

or install with ``setup.py``::

	$ cd python-mnist/
        $ python setup.py install

Code sample::

  from mnist import MNIST
  mndata = MNIST('./dir_with_mnist_data_files')
  images, labels = mndata.load_training()


Clean up directory
------------------

Delete all .pyc files with command::

	$ bash cleanup.sh
