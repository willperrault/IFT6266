import os
os.environ["THEANO_FLAGS"] = "floatX=float32"
import pickle as pkl
from six.moves import cPickle
import PIL.Image as Image
import timeit
import numpy as np
from matplotlib import pyplot as plt

import theano
import theano.tensor as T
import lasagne
import glob

import pdb


# Code structure inspired from https://lasagne.readthedocs.io/en/latest/user/tutorial.html

def build_CAEGAN(input_var=None):

    """Generator part of the Context Autoencoder GAN"""

    # Input is the a (Minibatch,3,64,64) matrix of images with no center
    # Output is a matrix of shape (Minibatch,3,32,32), the inpainting generated for the submitted images

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

def build_discriminator(input_var=None):

    """Discriminator part of the Context Autoencoder GAN"""

    # Input is the a (Minibatch,3,32,32) matrix of inpaitings
    # Output is a prediction bertween 0 and 1 of shape (Minibatch,1)

    network = lasagne.layers.InputLayer(shape=(None,3,32,32),input_var=input_var)

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

    network = lasagne.layers.DenseLayer(network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

def load_data(data_num=0, train=True):
    """ Function to load the either training or validation images from one of the 4 training files or 2 validation files (numbered respectively 0 to 3 or 0 and 1)"""

    if os.getcwd() == '/Users/williamperrault/Github/H2017/IFT6266/Code':
        if train:
            name = '/Users/williamperrault/Github/H2017/IFT6266/Data/train_data' + \
                str(data_num) + '.npy'
        else:
            name = '/Users/williamperrault/Github/H2017/IFT6266/Data/valid_data' + \
                str(data_num) + '.npy'
    else:
        if train:
            name = '/home2/ift6ed51/Data/train_data' + str(data_num) + '.npy'
        else:
            name = '/home2/ift6ed51/Data/valid_data' + str(data_num) + '.npy'
    f = open(name, 'rb')
    Data = np.load(f)

    f.close()

    return Data, Data.shape[0]

def produce_samples(params_gen,samples=3):
    """Function to produce samples from the generator"""
    # Input: params_gen is a string of the file name containing the saved parameters values for the generator that should be in the same working directory.
    # Samples is the number of images to generate and compare with the originals

    input_var = T.tensor4('input_var')
    model = build_CAEGAN(input_var)

    f = open(params_gen, 'rb')
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(model, params)

    generated_img = lasagne.layers.get_output(model)

    rand_int = np.random.choice(1, 1)
    data,data_size = load_data(rand_int[0],train=False)

    output_gen_fn = theano.function([input_var],generated_img,allow_input_downcast=True)

    rand_vect = np.random.choice(data_size,samples)
    fig = plt.figure()
    for count,i in enumerate(rand_vect):
        real = data[i,:,:,:] * 255.0
        real = np.swapaxes(real,0,1)
        real = np.swapaxes(real,1,2)

        image = data[i,:,:,:]
        fake = np.copy(image) * 255.0

        center = (int(np.floor(real.shape[0] / 2.)), int(np.floor(real.shape[1] / 2.)))

        image[:,center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0

        target = output_gen_fn(image.reshape((1,3,64,64)))[0,:,:,:] * 255.0

        fake[:,center[0]-16:center[0]+16, center[1]-16:center[1]+16] = target
        fake = np.swapaxes(fake,0,1)
        fake = np.swapaxes(fake,1,2)

        plt.axis('off')
        ax = plt.subplot(2,len(rand_vect),count+1)
        ax.imshow(real.astype("uint8"))
        plt.axis('off')
        ax1 = plt.subplot(2,len(rand_vect),(count+1)+len(rand_vect))
        ax1.imshow(fake.astype("uint8"))
    fig_name = 'CAEGAN2'+ params_gen[:-7] + params_gen[-2:]
    fig.savefig(fig_name)

def train_CAEGAN(learning_rate=0.01,n_epochs=30,batch_size=32):

    # Prepare Theano variables for inputs and targets
    context_image = T.tensor4('context_image')
    context_target = T.tensor4('context_target')

    generated_target = T.tensor4('generated_target')

    minibatch_index = T.iscalar()

    data1 = theano.shared(np.empty((1,1,1,1), dtype=theano.config.floatX),borrow=True)
    data2 = theano.shared(np.empty((1,1,1,1), dtype=theano.config.floatX),borrow=True)

    # Create neural network model
    print (".........building the model")

    generator = build_CAEGAN(context_image)
    generated_target = lasagne.layers.get_output(generator)

    discriminator = build_discriminator(context_target)

    prediction_real = lasagne.layers.get_output(discriminator)
    prediction_gen = lasagne.layers.get_output(discriminator,inputs=generated_target)

    # Calculate Loss :

    loss_gen_L2= T.mean(lasagne.objectives.squared_error(generated_target, context_target))
    loss_gen_GAN= 0.5*T.mean(lasagne.objectives.squared_error(prediction_gen, 1))

    loss_gen = 0.999*loss_gen_L2 + 0.001*loss_gen_GAN
    loss_gen = loss_gen.mean()

    loss_dis_GAN = 0.5*lasagne.objectives.squared_error(prediction_gen, 0) + 0.5*lasagne.objectives.squared_error(prediction_real, 1)
    loss_dis = 0.001*loss_dis_GAN.mean()

    params_gen = lasagne.layers.get_all_params(generator, trainable=True)
    params_dis = lasagne.layers.get_all_params(discriminator, trainable=True)

    updates_gen = lasagne.updates.adam(loss_gen, params_gen, learning_rate=0.0005, beta1=0.5, beta2=0.999, epsilon=1e-08)
    updates_dis = lasagne.updates.adam(loss_dis, params_dis, learning_rate=0.0004, beta1=0.5, beta2=0.999, epsilon=1e-08)

    # Compile

    train_gen_fn = theano.function([minibatch_index], loss_gen,allow_input_downcast=True, on_unused_input='ignore', updates=updates_gen,givens={context_image: data1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],context_target: data2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})

    train_dis_fn = theano.function([minibatch_index], loss_dis,allow_input_downcast=True,on_unused_input='ignore', updates=updates_dis,givens={context_target: data2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],context_image: data1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})

    # Training Loop

    print(".........begin Training")

    # The following code is heavily based if not directly from the MLP tutorial from http://deeplearning.net/tutorial/mlp.html

    best_iter = 0
    valid_score = 0

    epoch = 0
    done_looping = False

    while (epoch < n_epochs):

        w = open('CAEGAN_loss.txt', 'a')

        epoch = epoch + 1
        D_train_acc_cost = 0
        G_train_acc_cost = 0

        # Training loop:
        for data_split in range(4):

            temp,n_examples = load_data(data_split,train=True)

            center = (int(np.floor(temp.shape[2] / 2.)), int(np.floor(temp.shape[3] / 2.)))
            temp1 = np.copy(temp)
            temp1[:,:,center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
            temp2 = temp[:,:,center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

            data1.set_value(temp1)
            data2.set_value(temp2)

            n_train_batches = n_examples // batch_size

            for minibatch_index in range(n_train_batches):

                D_minibatch_avg_cost = train_dis_fn(minibatch_index)
                D_train_acc_cost += D_minibatch_avg_cost

                G_minibatch_avg_cost = train_gen_fn(minibatch_index)
                G_train_acc_cost += G_minibatch_avg_cost

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 2== 0:
                    w.write(str(D_minibatch_avg_cost)+' '+str(G_minibatch_avg_cost)+' \n')

        epoch_time = timeit.default_timer()
        print ("Training Epoch ran for : ", (epoch_time - start_time) / 60.)

        # Save parameters every 2 epochs
        if epoch % 2 == 1:
            generator_params = lasagne.layers.get_all_param_values(generator)
            discriminator_params = lasagne.layers.get_all_param_values(discriminator)

            f = open('CG_gen_params.save'+str(epoch), 'wb')
            cPickle.dump(generator_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

            g = open('CG_dis_params.save'+str(epoch), 'wb')
            cPickle.dump(discriminator_params, g, protocol=cPickle.HIGHEST_PROTOCOL)
            g.close()


        w.close()
    end_time = timeit.default_timer()

    print ("Code ran for : ", (end_time - start_time) / 60.)

if __name__ == '__main__':
    print ("Running Main")
    start_time = timeit.default_timer()
    train_CAEGAN()
    for i in range(1,2,2):
        produce_samples('CG_gen_params.save'+str(i))
