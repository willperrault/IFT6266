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

from utils import load_data
from utils import extract_target

# Code structure inspired from https://lasagne.readthedocs.io/en/latest/user/tutorial.html

def build_generator(input_var=None):
    """Generator part of the Context Autoencoder GAN"""

    # Input is the a (Minibatch,3,64,64) matrix of images with no center
    # Output is a matrix of shape (Minibatch,3,32,32), the inpainting generated for the submitted images

    network = lasagne.layers.InputLayer(shape=(None,3,64,64),input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network,num_filters=64,filter_size=(4,4), stride=(2,2))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=64,filter_size=(4,4), stride=(2,2),pad=2))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=128,filter_size=(4,4), stride=(2,2),pad=2))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=256,filter_size=(4,4), stride=(2,2),pad=1))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=1000,filter_size=(4,4), stride=(2,2)))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=256,filter_size=(4,4), stride=(2,2)))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=128,filter_size=(4,4), stride=(2,2),crop=1))

    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network,num_filters=64,filter_size=(4,4), stride=(2,2),crop=1))

    network = lasagne.layers.TransposedConv2DLayer(network,num_filters=3,filter_size=(4,4), stride=(2,2),crop=1,nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

def build_discriminator(input_var=None):
    """Discriminator part of the Context Autoencoder GAN"""

    # Input is the a (Minibatch,3,32,32) matrix of inpaitings
    # Output is a prediction bertween 0 and 1 of shape (Minibatch,1)

    network = lasagne.layers.InputLayer(shape=(None,3,32,32),input_var=input_var)

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=64,filter_size=(4,4), stride=(2,2),pad=1))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=128,filter_size=(4,4), stride=(2,2),pad=1))

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,num_filters=256,filter_size=(4,4), stride=(2,2),pad=1))

    network = lasagne.layers.DenseLayer(network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

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

def train_CAEGAN(learning_rateG=0.004,learning_rateD=0.005,n_epochs=1,batch_size=32):

    # Prepare Theano variables for inputs and targets
    input_image = T.tensor4('input_image',dtype=theano.config.floatX)
    gen_output = T.tensor4('gen_output',dtype=theano.config.floatX)
    dis_output = T.matrix('dis_output',dtype=theano.config.floatX)
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

    generator = build_generator(input_image)
    discriminator = build_discriminator(target_image)

        # with open('CG_gen_params_best.save15', 'rb') as fp:
        #     params = cPickle.load(fp)
        #     lasagne.layers.set_all_param_values(generator, params)
        #
        # with open('CG_dis_params_best.save15', 'rb') as fp:
        #     params = cPickle.load(fp)
        #     lasagne.layers.set_all_param_values(discriminator, params)

    gen_output = lasagne.layers.get_output(generator)
    dis_output = lasagne.layers.get_output(discriminator)
    generated_img = lasagne.layers.get_output(generator, inputs=sample_context)
    dis_output_fake = lasagne.layers.get_output(discriminator, inputs=gen_output)

    # Calculate Loss :

    loss_gen_L2= T.mean(lasagne.objectives.squared_error(gen_output, target_image))
    loss_gen_GAN= T.mean(lasagne.objectives.binary_crossentropy(dis_output_fake, 1))

    loss_gen = 0.999*loss_gen_L2 + 0.001*loss_gen_GAN

    loss_dis_GAN = T.mean(lasagne.objectives.binary_crossentropy(dis_output_fake, 0) + lasagne.objectives.binary_crossentropy(dis_output, 1))

    loss_dis = 0.001*loss_dis_GAN

    params_gen = lasagne.layers.get_all_params(generator, trainable=True)
    params_dis = lasagne.layers.get_all_params(discriminator, trainable=True)

    updates_gen = lasagne.updates.adam(loss_gen, params_gen, learning_rate=learning_rateG)
    updates_dis = lasagne.updates.adam(loss_dis, params_dis, learning_rate=learning_rateD)

    # Compile functions for training
    train_gen_fn = theano.function([minibatch_index], loss_gen,allow_input_downcast=True, on_unused_input='ignore', updates=updates_gen,givens={input_image: shared_input[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],target_image: shared_target[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})

    train_dis_fn = theano.function([minibatch_index], loss_dis,allow_input_downcast=True,on_unused_input='ignore', updates=updates_dis,givens={input_image: shared_input[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],target_image: shared_target[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})

    output_gen_fn = theano.function(
        [sample_context], generated_img, allow_input_downcast=True)

    # Training Loop

    print(".........begin Training")

    # The following code is heavily based if not directly from the MLP tutorial from http://deeplearning.net/tutorial/mlp.html

    best_iter = 0
    best_valid_score = np.inf

    epoch = 0
    done_looping = False

    detailed_training_loss1 = []
    detailed_training_loss2 = []
    training_loss1 = []
    training_loss2 = []

    while (epoch < n_epochs):

        w = open('CAEGAN_loss.txt', 'a')

        epoch = epoch + 1
        temp_loss1 = []
        temp_loss2 = []

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

                minibatch_cost1 = train_gen_fn(minibatch_index)
                temp_loss1.append(minibatch_cost1)

                minibatch_cost2 = train_dis_fn(minibatch_index)
                temp_loss2.append(minibatch_cost2)

                if minibatch_index % 20 == 0:
                    detailed_training_loss1.append(minibatch_cost1)
                    detailed_training_loss2.append(minibatch_cost2)

        # Take 3 random succesive images from the last batch to generate samples
        rand_int = np.random.randint(0, high=n_examples-3)
        sample_context = data[rand_int:rand_int+3,:,:,:]

        epoch_examples = output_gen_fn(sample_context) * 255.0
        epoch_examples = np.swapaxes(epoch_examples, 1, 2)
        epoch_examples = np.swapaxes(epoch_examples, 2, 3)
        sample_context = np.swapaxes(sample_context, 1, 2)
        sample_context = np.swapaxes(sample_context, 2, 3)

        np.save('CG_samples'+str(epoch),epoch_examples)
        np.save('CG_samples_context'+str(epoch),sample_context*255.0)

        # Keep relevant loss information
        epoch_mean = sum(temp_loss1)/len(temp_loss1)
        training_loss1.append(epoch_mean)

        epoch_mean = sum(temp_loss2)/len(temp_loss2)
        training_loss2.append(epoch_mean)

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

    with open('CG_detailed_training_loss_gen.save', 'wb') as fp:
        pkl.dump(detailed_training_loss1, fp)

    with open('CG_detailed_training_loss_dis.save', 'wb') as fp:
        pkl.dump(detailed_training_loss2, fp)

    with open('CG_training_loss_gen.save', 'wb') as fp:
        pkl.dump(training_loss1, fp)

    with open('CG_training_loss_dis.save', 'wb') as fp:
        pkl.dump(training_loss2, fp)

    end_time = timeit.default_timer()

    print ("Code ran for : ", (end_time - start_time) / 60.)

if __name__ == '__main__':
    print ("Running Main")
    start_time = timeit.default_timer()
    train_CAEGAN()
