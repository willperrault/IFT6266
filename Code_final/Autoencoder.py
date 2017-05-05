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

from utils import load_data
from utils import extract_target

# Code structure inspired from
# https://lasagne.readthedocs.io/en/latest/user/tutorial.html


def build_autoencoder(input_var=None):
    """Creates the sequence of layers that make up the autoencoder """

    # Input is the a (Minibatch,3,64,64) matrix of images with no center
    # Output is a matrix of shape (Minibatch,3,32,32), the inpainting generated for the submitted images

    autoencoder = lasagne.layers.InputLayer(
        shape=(None, 3, 64, 64), input_var=input_var)

    autoencoder = lasagne.layers.Conv2DLayer(autoencoder, num_filters=32, filter_size=(
        4, 4), stride=(2, 2), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(autoencoder, num_filters=64, filter_size=(
        4, 4), stride=(2, 2), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2)))

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(autoencoder, num_filters=128, filter_size=(
        5, 5), stride=(2, 2), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2)))

    autoencoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(autoencoder, num_filters=256, filter_size=(
        5, 5), stride=(2, 2), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2)))

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

    return autoencoder

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

def train_AE(learning_rate=0.0009, n_epochs=5, batch_size=128):

    # Prepare Theano variables for inputs and targets
    input_image = T.tensor4('input_image',dtype=theano.config.floatX)
    ae_output = T.tensor4('ae_output',dtype=theano.config.floatX)
    target_image = T.tensor4('target_image',dtype=theano.config.floatX)
    sample_context = T.tensor4('target_image',dtype=theano.config.floatX)

    minibatch_index = T.iscalar('minibatch_index')

    # Prepare shared variable for use by GPU
    shared_input = theano.shared(
        np.empty((10348, 3, 64, 64), dtype=theano.config.floatX), borrow=True)

    shared_target = theano.shared(
        np.empty((10348, 3, 64, 64), dtype=theano.config.floatX), borrow=True)

    # Create neural network model
    print (".........building the model")

    autoencoder = build_autoencoder(input_image)
    ae_output = lasagne.layers.get_output(autoencoder)
    generated_img = lasagne.layers.get_output(autoencoder, inputs=sample_context)

    # Calculate Loss:
    loss_ae = T.mean(lasagne.objectives.squared_error(ae_output, target_image))

    # Extract parameters and prepare updates
    params_ae = lasagne.layers.get_all_params(autoencoder, trainable=True)

    updates_ae = lasagne.updates.adam(
        loss_ae, params_ae, learning_rate=learning_rate)

    # Compile functions for training
    train_fn = theano.function([minibatch_index], loss_ae, allow_input_downcast=True, updates=updates_ae, givens={input_image: shared_input[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], target_image: shared_target[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})

    valid_fn = theano.function([minibatch_index], loss_ae, allow_input_downcast=True, givens={input_image: shared_input[
                                  minibatch_index * batch_size: (minibatch_index + 1) * batch_size], target_image: shared_target[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})

    output_ae_fn = theano.function(
        [sample_context], generated_img, allow_input_downcast=True)

    # Training Loop

    print(".........begin training")

    # The following code is heavily based if not directly from the MLP
    # tutorial from http://deeplearning.net/tutorial/mlp.html

    best_iter = 0
    best_valid_score = np.inf

    epoch = 0
    done_looping = False

    detailed_training_loss = []
    training_loss = []
    valid_loss = []

    while (epoch < n_epochs):

        epoch = epoch + 1
        temp_loss = []

        # Training loop
        for data_split in range(8):

            # Load original images and their number from the loaded file
            data, n_examples = load_data(data_split, train=True)

            # Split the image in context and target
            context,targets = extract_target(data)

            # Load context and target in shared variables
            shared_input.set_value(context)
            shared_target.set_value(targets)

            n_train_batches = n_examples // batch_size

            for minibatch_index in range(n_train_batches):

                minibatch_cost = train_fn(minibatch_index)
                temp_loss.append(minibatch_cost)

                if minibatch_index % 20 == 0:
                    detailed_training_loss.append(minibatch_cost)

        # Take 3 random succesive images from the last batch to generate samples
        rand_int = np.random.randint(0, high=n_examples-3)
        sample_context = data[rand_int:rand_int+3,:,:,:]

        epoch_examples = output_ae_fn(sample_context) * 255.0
        epoch_examples = np.swapaxes(epoch_examples, 1, 2)
        epoch_examples = np.swapaxes(epoch_examples, 2, 3)
        sample_context = np.swapaxes(sample_context, 1, 2)
        sample_context = np.swapaxes(sample_context, 2, 3)

        np.save('AE_samples'+str(epoch),epoch_examples)
        np.save('AE_samples_context'+str(epoch),sample_context*255.0)

        # Keep relevant loss information
        epoch_mean = sum(temp_loss)/len(temp_loss)
        training_loss.append(epoch_mean)

        epoch_time = timeit.default_timer()
        print ("Training Epoch ran for : ", (epoch_time - start_time) / 60.)

        # Valid loop:
        temp_loss = []

        for data_split in range(4):

            # Load original images and their number from the loaded file
            data, n_examples = load_data(data_split, train=False)

            # Split the image in context and target
            context,targets = extract_target(data)

            # Load context and target in shared variables
            shared_input.set_value(context)
            shared_target.set_value(targets)

            n_train_batches = n_examples // batch_size

            for minibatch_index in range(n_train_batches):

                minibatch_cost = train_fn(minibatch_index)
                temp_loss.append(minibatch_cost)

        # Keep relevant loss information
        epoch_mean = sum(temp_loss)/len(temp_loss)
        valid_loss.append(epoch_mean)

        epoch_time = timeit.default_timer()
        print ("Valid Epoch ran for : ", (epoch_time - start_time) / 60.)

        # Save parameters every 3 epochs or when new best
        if valid_loss[-1] < best_valid_score:
            autoencoder_params = lasagne.layers.get_all_param_values(
                autoencoder)

            f = open('AE_params_best.save', 'wb')
            cPickle.dump(autoencoder_params, f,
                         protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

            best_valid_score = valid_loss[-1]
            best_iter = epoch

        elif epoch % 3 == 1:
            autoencoder_params = lasagne.layers.get_all_param_values(
                autoencoder)

            f = open('AE_params' + str(epoch) + 'save', 'wb')
            cPickle.dump(autoencoder_params, f,
                         protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

    # Save loss information for later analysis:
    with open('AE_detailed_training_loss.save', 'wb') as fp:
        pkl.dump(detailed_training_loss, fp)

    with open('AE_valid_loss.save', 'wb') as fp:
        pkl.dump(valid_loss, fp)

    with open('AE_training_loss.save', 'wb') as fp:
        pkl.dump(training_loss, fp)

    end_time = timeit.default_timer()

    print ("Code ran for : ", (end_time - start_time) / 60.)

    print('Best valid score of %f at epoch %i' % (best_valid_score,epoch))

if __name__ == '__main__':
    print ("Running Main")
    start_time = timeit.default_timer()
    train_AE()
