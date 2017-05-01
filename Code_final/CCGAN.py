import os
os.environ["THEANO_FLAGS"] = "floatX=float32"
import pickle as pkl
from six.moves import cPickle
import timeit
import numpy as np

import theano
import theano.tensor as T
import lasagne
import glob

import pdb

# This script is a draft of a model that wasn't implemented

# Code structure inspired from https://lasagne.readthedocs.io/en/latest/user/tutorial.html

def build_generator(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None,3,64,64),input_var=input_var)

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=128,filter_size=(4,4), stride=(2,2)))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=256,filter_size=(4,4), stride=(2,2),pad=2))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=512,filter_size=(4,4), stride=(2,2),pad=1))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=256,filter_size=(4,4), stride=(2,2),crop=1))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=128,filter_size=(4,4), stride=(2,2),crop=1))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=3,filter_size=(4,4), stride=(2,2),crop=1))

    return network

def build_discriminator(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None,3,64,64),input_var=input_var)

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=64,filter_size=(5,5), stride=(2,2)))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=128,filter_size=(5,5), stride=(2,2)))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=256,filter_size=(5,5), stride=(2,2)))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=512,filter_size=(5,5), stride=(2,2)))

    network = lasagne.layers.DenseLayer(network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return network
