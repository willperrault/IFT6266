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

def generate_noise(batch_size,noise_length=100):
    """Generate noise for the GAN"""

    noise = np.random.normal(loc=0.0, scale=1.0, size=(batch_size,noise_length,1,1))

    return noise

def build_generator(input_var=None):
    """Creates the sequence of layers that make up the generator for the GAN"""

    # Input is noise with shape (Minibatch,100)injected to generate an image
    # Output is a (Minibatch,3,64,64)

    generator = lasagne.layers.InputLayer(shape=(None,100,1,1),input_var=input_var)

    generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=512,filter_size=(4,4), stride=(1,1)))

    generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=256,filter_size=(5,5), stride=(1,1)))

    generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=128,filter_size=(4,4), stride=(2,2),crop=1))

    generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=64,filter_size=(6,6), stride=(2,2),crop=2))

    generator = lasagne.layers.TransposedConv2DLayer(generator,num_filters=3,filter_size=(6,6), stride=(2,2),crop=2)

    return generator

def build_discriminator(input_var=None):

    """Creates the sequence of layers that make up the discriminator for the GAN"""

    # Input is a matix of image of shape (Minibatch,3,64,64)
    # Output a matrix of predictions between 0 and 1 of shape (Minibatch,1)

    discriminator = lasagne.layers.InputLayer(shape=(None,3,64,64),input_var=input_var)

    discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=64,filter_size=(5,5), stride=(2,2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))

    discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=128,filter_size=(5,5), stride=(2,2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))

    discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=256,filter_size=(5,5), stride=(2,2), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))

    discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=100,filter_size=(5,5), stride=(2,2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))

    # Output layer:
    discriminator = lasagne.layers.DenseLayer(discriminator, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return discriminator

def produce_samples(params_gen,samples=3):
    """Function to produce samples from the generator"""
    # Input: params_gen is a string of the file name containing the saved parameters values for the generator that should be in the same working directory.
    # Samples is the number of images to generate and compare with the originals

    noise = T.tensor4('noise')

    # Build model and load parameters from the file
    generator = build_generator(noise)
    f = open(params_gen, 'rb')
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(generator, params)

    # Create variable for output of the generator
    generated_img = lasagne.layers.get_output(generator)

    #
    rand_int = np.random.choice(3, 1)
    data,data_size = load_data(rand_int[0])

    output_gen_fn = theano.function([noise],generated_img,allow_input_downcast=True)

    rand_vect = np.random.choice(data_size,samples)
    fig = plt.figure()
    for count,i in enumerate(rand_vect):
        real = data[i,:,:,:] * 255.0
        real = np.swapaxes(real,0,1)
        real = np.swapaxes(real,1,2)

        noise = generate_noise(batch_size=1,noise_length=100)
        generated = output_gen_fn(noise)[0,:,:,:] * 255.0
        generated = np.swapaxes(generated,0,1)
        generated = np.swapaxes(generated,1,2)

        center = (int(np.floor(real.shape[0] / 2.)), int(np.floor(real.shape[1] / 2.)))
        fake = np.copy(real)
        target = generated[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        fake[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = target

        plt.axis('off')
        ax = plt.subplot(2,len(rand_vect),count+1)
        ax.imshow(real.astype("uint8"))
        plt.axis('off')
        ax1 = plt.subplot(2,len(rand_vect),(count+1)+len(rand_vect))
        ax1.imshow(fake.astype("uint8"))
    fig_name = 'DCGAN_2_'+ params_gen[:-6] + params_gen[-1]
    fig.savefig(fig_name)

def load_data(data_num=0):
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

def train_DCGAN(learning_rate=0.01,n_epochs=20,batch_size=32):

    # Prepare Theano variables for inputs and targets
    noise = T.tensor4('noise')
    real_image = T.tensor4('real_image')
    minibatch_index = T.iscalar()
    data = theano.shared(np.empty((1,1,1,1), dtype=theano.config.floatX),borrow=True)

    # Create neural network model
    print (".........building the model")

    generator = build_generator(noise)
    gen_image = lasagne.layers.get_output(generator)

    discriminator = build_discriminator(gen_image)

    prediction_gen = lasagne.layers.get_output(discriminator)
    prediction_real = lasagne.layers.get_output(discriminator,inputs=real_image)

    # Calculate Loss :
    #loss_gen = lasagne.objectives.binary_crossentropy(prediction_gen,0)
    loss_gen = 0.5*lasagne.objectives.squared_error(prediction_gen, 1)
    loss_gen = loss_gen.mean()

    #loss_dis = lasagne.objectives.binary_crossentropy(prediction_real,1) + lasagne.objectives.binary_crossentropy(prediction_gen,0)
    loss_dis = 0.5*lasagne.objectives.squared_error(prediction_gen, 0) + 0.5*lasagne.objectives.squared_error(prediction_real, 1)
    loss_dis = loss_dis.mean()

    params_gen = lasagne.layers.get_all_params(generator, trainable=True)
    params_dis = lasagne.layers.get_all_params(discriminator, trainable=True)

    updates_gen = lasagne.updates.adam(loss_gen, params_gen, learning_rate=0.0002, beta1=0.5, beta2=0.999, epsilon=1e-08)
    updates_dis = lasagne.updates.adam(loss_dis, params_dis, learning_rate=0.0002, beta1=0.5, beta2=0.999, epsilon=1e-08)

    # Compile
    train_gen_fn = theano.function([noise], loss_gen,allow_input_downcast=True, updates=updates_gen)

    train_dis_fn = theano.function([noise, minibatch_index], loss_dis,allow_input_downcast=True,on_unused_input='ignore', updates=updates_dis,givens={real_image: data[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})

    # Training Loop

    print(".........begin Training")

    # The following code is heavily based if not directly from the MLP tutorial from http://deeplearning.net/tutorial/mlp.html

    best_iter = 0
    test_score = 0

    epoch = 0
    done_looping = False

    w = open('gen_loss.txt', 'a')
    x = open('dis_loss.txt','a')

    while (epoch < n_epochs):
        epoch = epoch + 1
        for data_split in range(4):
            temp,n_examples = load_data(data_split)
            data.set_value(temp)

            n_train_batches = n_examples // batch_size

            for minibatch_index in range(n_train_batches):

                noise = generate_noise(batch_size=batch_size,noise_length=100)
                minibatch_avg_cost_dis = train_dis_fn(noise,minibatch_index)

                if minibatch_index % 20==0:
                    for _ in range(10):
                        minibatch_avg_cost_gen = train_gen_fn(noise)
                        noise = generate_noise(batch_size=batch_size,noise_length=100)

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    w.write(str(minibatch_avg_cost_gen)+' \n')
                    x.write(str(minibatch_avg_cost_dis)+' \n')

            epoch_time = timeit.default_timer()
            print ("Code ran for : ", (epoch_time - start_time) / 60.)

            # Save parameters every 2 epochs
            if epoch % 2 == 1:
                generator_params = lasagne.layers.get_all_param_values(generator)
                discriminator_params = lasagne.layers.get_all_param_values(discriminator)

                f = open('gen_params.save'+str(epoch), 'wb')
                cPickle.dump(generator_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()

                g = open('dis_params.save'+str(epoch), 'wb')
                cPickle.dump(discriminator_params, g, protocol=cPickle.HIGHEST_PROTOCOL)
                g.close()

    w.close()
    x.close()
    end_time = timeit.default_timer()

    print ("Code ran for : ", (end_time - start_time) / 60.)

if __name__ == '__main__':
    print ("Running Main")
    start_time = timeit.default_timer()
    train_DCGAN()
    produce_samples('gen_params.save7')
