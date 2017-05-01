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


# Code structure inspired from
# https://lasagne.readthedocs.io/en/latest/user/tutorial.html

def build_autoencoder(input_var=None):
    """Creates the sequence of layers that make up the autoencoder """

    # Input is the a (Minibatch,3,64,64) matrix of images with no center
    # Output is a matrix of shape (Minibatch,3,32,32), the inpainting generated for the submitted images

    autoencoder = lasagne.layers.InputLayer(
        shape=(None, 3, 64, 64), input_var=input_var)

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
        autoencoder, num_filters=32, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2), pad=2))

    autoencoder = lasagne.layers.Pool2DLayer(
        autoencoder, (2, 2), mode='average_exc_pad')

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
        autoencoder, num_filters=64, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2), pad=2))

    autoencoder = lasagne.layers.Pool2DLayer(
        autoencoder, (2, 2), mode='average_exc_pad')

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
        autoencoder, num_filters=128, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2)))

    autoencoder = lasagne.layers.Pool2DLayer(
        autoencoder, (2, 2), mode='average_exc_pad')

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
        autoencoder, num_filters=256, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2)))

    autoencoder = lasagne.layers.Pool2DLayer(
        autoencoder, (2, 2), mode='average_exc_pad')

    autoencoder = lasagne.layers.DenseLayer(
        autoencoder, num_units=500, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))

    autoencoder = lasagne.layers.DenseLayer(
        autoencoder, num_units=500, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))

    autoencoder = lasagne.layers.ReshapeLayer(autoencoder, (-1, 500, 1, 1))

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(
        autoencoder, num_filters=256, filter_size=(4, 4), stride=(2, 2), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2)))

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(autoencoder, num_filters=128, filter_size=(
        4, 4), stride=(2, 2), crop=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2)))

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(autoencoder, num_filters=64, filter_size=(
        4, 4), stride=(2, 2), crop=1, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2)))

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(
        autoencoder, num_filters=3, filter_size=(4, 4), stride=(2, 2), crop=1, nonlinearity=lasagne.nonlinearities.sigmoid))
    #
    return autoencoder

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

def produce_samples(params_gen, samples=3):
    """Function to produce samples from the generator"""
    # Input: params_gen is a string of the file name containing the saved parameters values for the generator that should be in the same working directory.
    # Samples is the number of images to generate and compare with the originals

    input_var = T.tensor4('input_var')

    # Build model and Load parameters:
    autoencoder = build_autoencoder(input_var)
    f = open(params_gen, 'rb')
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(autoencoder, params)

    # Network output:
    generated_img = lasagne.layers.get_output(autoencoder, inputs=input_var)

    rand_int = np.random.choice(1, 1)
    data, data_size = load_data(rand_int[0], train=False)

    output_gen_fn = theano.function(
        [input_var], generated_img, allow_input_downcast=True)

    rand_vect = np.random.choice(data_size, samples)
    fig = plt.figure()
    for count, i in enumerate(rand_vect):
        real = data[i, :, :, :] * 255.0
        real = np.swapaxes(real, 0, 1)
        real = np.swapaxes(real, 1, 2)

        image = data[i, :, :, :]
        fake = np.copy(image) * 255.0

        center = (int(np.floor(real.shape[0] / 2.)),
                  int(np.floor(real.shape[1] / 2.)))

        image[:, center[0] - 16:center[0] + 16,
              center[1] - 16:center[1] + 16] = 0

        target = output_gen_fn(image.reshape((1, 3, 64, 64)))[
            0, :, :, :] * 255.0

        fake[:, center[0] - 16:center[0] + 16,
             center[1] - 16:center[1] + 16] = target
        fake = np.swapaxes(fake, 0, 1)
        fake = np.swapaxes(fake, 1, 2)

        plt.axis('off')
        ax = plt.subplot(2, len(rand_vect), count + 1)
        ax.imshow(real.astype("uint8"))
        plt.axis('off')
        ax1 = plt.subplot(2, len(rand_vect), (count + 1) + len(rand_vect))
        ax1.imshow(fake.astype("uint8"))

    fig_name = 'AE5' + params_gen[:-7] + params_gen[-2:]
    fig.savefig(fig_name)

def train_AE(learning_rate=0.01, n_epochs=10, batch_size=128):

    # Prepare Theano variables for inputs and targets
    real_image = T.tensor4('real_image')
    target = T.tensor4('target')
    minibatch_index = T.iscalar()

    data1 = theano.shared(
        np.empty((1, 1, 1, 1), dtype=theano.config.floatX), borrow=True)
    data2 = theano.shared(
        np.empty((1, 1, 1, 1), dtype=theano.config.floatX), borrow=True)

    # Create neural network model
    print (".........building the model")

    autoencoder = build_autoencoder(real_image)
    gen_image = lasagne.layers.get_output(autoencoder)

    # Calculate Loss :
    loss_ae = lasagne.objectives.squared_error(gen_image, target)
    loss_ae = loss_ae.mean()

    params_ae = lasagne.layers.get_all_params(autoencoder, trainable=True)

    updates_ae = lasagne.updates.adam(
        loss_ae, params_ae, learning_rate=0.001, beta1=0.5, beta2=0.999, epsilon=1e-08)

    # Compile
    train_fn = theano.function([minibatch_index], loss_ae, allow_input_downcast=True, updates=updates_ae, givens={real_image: data1[
                               minibatch_index * batch_size: (minibatch_index + 1) * batch_size], target: data2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})

    valid_ae_fn = theano.function([minibatch_index], loss_ae, allow_input_downcast=True, givens={real_image: data1[
                                  minibatch_index * batch_size: (minibatch_index + 1) * batch_size], target: data2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})

    # Training Loop

    print(".........begin Training")

    # The following code is heavily based if not directly from the MLP
    # tutorial from http://deeplearning.net/tutorial/mlp.html

    best_iter = 0
    valid_score = 0

    epoch = 0
    done_looping = False

    while (epoch < n_epochs):
        w = open('ae2_loss.txt', 'a')
        epoch = epoch + 1
        train_acc_cost = 0

        #Training loop:
        for data_split in range(4):
            temp, n_examples = load_data(data_split, train=True)

            center = (
                int(np.floor(temp.shape[2] / 2.)), int(np.floor(temp.shape[3] / 2.)))
            temp1 = np.copy(temp)
            temp1[:, :, center[0] - 16:center[0] +
                  16, center[1] - 16:center[1] + 16] = 0
            temp2 = temp[:, :, center[0] - 16:center[0] +
                         16, center[1] - 16:center[1] + 16]

            data1.set_value(temp1)
            data2.set_value(temp2)

            n_train_batches = n_examples // batch_size

            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost_ae = train_ae_fn(minibatch_index)
                train_acc_cost += minibatch_avg_cost_ae

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if False:  # iter % 5 == 0:
                    w.write(str(minibatch_avg_cost_ae) + ' \n')

        train_score = train_acc_cost / n_train_batches
        epoch_time = timeit.default_timer()
        print ("Training Epoch ran for : ", (epoch_time - start_time) / 60.)

        # Valid loop:
        valid_acc_cost = 0
        for data_split in range(2):
            temp, n_examples = load_data(data_split, train=False)

            center = (
                int(np.floor(temp.shape[2] / 2.)), int(np.floor(temp.shape[3] / 2.)))
            temp1 = np.copy(temp)
            temp1[:, :, center[0] - 16:center[0] +
                  16, center[1] - 16:center[1] + 16] = 0
            temp2 = temp[:, :, center[0] - 16:center[0] +
                         16, center[1] - 16:center[1] + 16]

            data1.set_value(temp1)
            data2.set_value(temp2)

            n_train_batches = n_examples // batch_size

            for minibatch_index in range(n_train_batches):

                valid_acc_cost += valid_ae_fn(minibatch_index)

        valid_score = valid_acc_cost / n_train_batches

        w.write(str(train_score) + ' ' + str(valid_score) + ' \n')

        valid_time = timeit.default_timer()
        print ("Valid Epoch ran for : ", (valid_time - start_time) / 60.)

        # Save parameters every 2 epochs
        if epoch % 2 == 1:
            autoencoder_params = lasagne.layers.get_all_param_values(
                autoencoder)

            f = open('ae2_params.save' + str(epoch), 'wb')
            cPickle.dump(autoencoder_params, f,
                         protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        w.close()
    end_time = timeit.default_timer()

    print ("Code ran for : ", (end_time - start_time) / 60.)

if __name__ == '__main__':
    print ("Running Main")
    start_time = timeit.default_timer()
    train_AE()
    for i in range(1,23,2):
        produce_samples('ae2_params.save'+str(i))
