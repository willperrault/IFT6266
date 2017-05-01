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

# Script used to build model layer by layers by checking output etc..

def build_network(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None,3,64,64),input_var=input_var)

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=128,filter_size=(4,4), stride=(2,2)))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=256,filter_size=(4,4), stride=(2,2),pad=2))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=512,filter_size=(4,4), stride=(2,2),pad=1))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=256,filter_size=(4,4), stride=(2,2),crop=1))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=128,filter_size=(4,4), stride=(2,2),crop=1))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=3,filter_size=(4,4), stride=(2,2),crop=1))

    return network

def build_network2(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None,3,64,64),input_var=input_var)

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2), pad=2))

    network = lasagne.layers.Pool2DLayer(
        network, (2, 2), mode='average_exc_pad')

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
        network, num_filters=128, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2), pad=2))

    network = lasagne.layers.Pool2DLayer(
        network, (2, 2), mode='average_exc_pad')

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
        network, num_filters=256, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2), pad=2))

    network = lasagne.layers.Pool2DLayer(
        network, (2, 2), mode='average_exc_pad')

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=512, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2), pad=2))

    network = lasagne.layers.Pool2DLayer(
            network, (2, 2), mode='average_exc_pad')

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
        network, num_filters=1000, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2), pad=2))

    network = lasagne.layers.Pool2DLayer(
        network, (3, 3), mode='average_exc_pad')

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=512,filter_size=(4,4), stride=(2,2)))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=256,filter_size=(4,4), stride=(2,2),crop=1))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=128,filter_size=(4,4), stride=(2,2),crop=1))

    network = lasagne.layers.TransposedConv2DLayer(network,num_filters=3,filter_size=(4,4), stride=(2,2),crop=1,nonlinearity=lasagne.nonlinearities.sigmoid)

    return network


if True:
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    image = T.tensor4('image')
    real_image = T.tensor4('real_image')

    target_var = T.ivector('targets')
    # Create neural network model
    network = build_network2(input_var)
    image = lasagne.layers.get_output(network)

    prediction_real = lasagne.layers.get_output(network,inputs=real_image)

    loss_nw = lasagne.objectives.squared_error(image,real_image)
    loss_nw = loss_nw.mean()

    train_auto_fn = theano.function([input_var, real_image], loss_nw,allow_input_downcast=True)

    input_var =  np.random.normal(loc=0.0, scale=1.0, size=(1,3,64,64))
    real_image =  np.random.normal(loc=0.0, scale=1.0, size=(1,3,32,32))

    # loss = train_auto_fn(input_var.astype('float32'),real_image.astype('float32'))

    image = lasagne.layers.get_output(network,inputs=input_var.astype('float32'))
    print(image.shape.eval())

    # print(loss)

    # temp,n_examples = load_data(0,train=False)
    #
    # center = (int(np.floor(temp.shape[2] / 2.)), int(np.floor(temp.shape[3] / 2.)))
    # temp1 = np.copy(temp)
    # temp1[:,:,center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
    # temp2 = temp[:,:,center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

    # print(temp.shape)
    # print(temp1.shape)
    # print(temp2.shape)
