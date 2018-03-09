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
import os
import shutil

from SpikingConvNet import utils
from SpikingConvNet.parameters import *
from SpikingConvNet.classes import ConvLayer, Classifier, SpikingModel

import spynnaker8 as s
# import pyNN.utility.plotting as plot
from training import training_function
from testing import testing_function
from info import info
from multi import parallel_populations

parser = argparse.ArgumentParser(description='Deep Spiking CNN with STDP Rule')

parser.add_argument(
    '--debug',
    action="store_true",
    help='Print debug information in console (default: False)')

parser.add_argument(
    '--no-backup',
    action="store_true",
    help='Backup files and information in separate directory (default: False)')

parser.add_argument(
    '--mode',
    required= True,
    metavar='<training, testing, loaddata>',
    help='Mode of execution')

parser.add_argument(
    '--layer',
    metavar='<1,2,svm>',
    default= "None",
    help='Specify layer to train (required for training mode)')



def build_model(rc):

    model = SpikingModel(input_tensor=(28,28,1), run_control=rc)
    model.add(ConvLayer(4, shape=(4,4), stride=3))
    # model.add(ConvLayer(4,shape=(2,2), stride=1))
    model.add(Classifier())

    return model

if __name__ == '__main__':
    args = parser.parse_args()
    rc = utils.RunControl(args=args)
    model = build_model(rc)
    model.print_structure()

    # set up control settings for program
    rc = utils.RunControl(args=args)

    # build backup structure
    if rc.backup:
        try:
            os.makedirs(rc.backup_path)
        except OSError:
            if not os.path.isdir(rc.backup_path):
                raise NameError("Backup path is not a directory")
        try:
            shutil.copy2(PACKAGE+'parameters.py', rc.backup_path+'parameters.py')
        except:
            raise NameError("Please switch current Path to main.py directory")
    else:       # deny backups
        rc.backup_path = None

    if args.mode == 'training':
        if args.layer == "None":
            raise NameError("Define layer to train with suffix --layer")
        elif args.layer == "svm":
            rc.train_svm = True
            rc.train_layer = 0
            rc.rebuild = True
        elif eval(args.layer) > model.number_layers:
            raise NameError("Appreciated layer is not defined in model")
        else:
            rc.train_layer = eval(args.layer)
        training_function(rc, model)

    elif args.mode == 'multi':
        if args.layer == "None":
            raise NameError("Define layer to train with suffix --layer")
        rc.train_layer=eval(args.layer)
        parallel_populations(rc)

    elif args.mode == 'testing':
        rc.rebuild = True
        testing_function(rc, model )

    elif args.mode == 'loaddata':
        print("Loading Train Data - {} examples per class".format(TRAIN_EXAMPLES))
        utils.load_MNIST_digits_shuffled(rc, mode="train_examples")
        print("Loading Test Data - {} examples per class".format(TEST_EXAMPLES))
        utils.load_MNIST_digits_shuffled(rc, mode="test_examples")
        if rc.backup:
            try:
                shutil.copy2(DATA_PATH+'X_train', rc.backup_path+'X_train')
                shutil.copy2(DATA_PATH+'y_train', rc.backup_path+'y_train')
                shutil.copy2(DATA_PATH+'X_test', rc.backup_path+'X_test')
                shutil.copy2(DATA_PATH+'y_test', rc.backup_path+'y_test')
                print("Successfully backed up data")
            except:
                print("Backup of Test/Train data failed")
                rc.logging.error("Backup of Test/Train data failed")

    elif args.mode == 'info':
        info(rc, model)
    else:
        print("Mode not available. Check main.py -h")

    if rc.backup:   # backup logfile
        shutil.copy2(LOGFILE, rc.backup_path+LOGFILE)
