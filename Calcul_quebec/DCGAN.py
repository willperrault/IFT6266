import os
import pickle as pkl
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

    noise = np.random.normal(loc=0.0, scale=1.0, size=(batch_size,noise_length,1,1))

    return noise

def build_generator(input_var=None):
    # Input will be noise injected to generate an image

    # Input will be noise injected to generate an image

    generator = lasagne.layers.InputLayer(shape=(None,100,1,1),input_var=input_var)

    generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=256,filter_size=(4,4), stride=(1,1)))

    generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=128,filter_size=(5,5), stride=(1,1)))

    generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=64,filter_size=(4,4), stride=(2,2),crop=1))

    generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=32,filter_size=(6,6), stride=(2,2),crop=2))

    generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=3,filter_size=(6,6), stride=(2,2),crop=2))

    return generator

def build_discriminator(input_var=None):

    discriminator = lasagne.layers.InputLayer(shape=(None,3,64,64),input_var=input_var)

    discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=64,filter_size=(5,5), stride=(2,2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))

    discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=128,filter_size=(5,5), stride=(2,2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))

    discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=256,filter_size=(5,5), stride=(2,2), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))

    discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=100,filter_size=(5,5), stride=(2,2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))

    # Output layer:
    discriminator = lasagne.layers.DenseLayer(discriminator, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return discriminator

    # lasagne.layers.get_output_shape(generator)

def cost():
    #??????
    pass

def load_data(split="inpainting/train2014", caption_path="inpainting/dict_key_imgID_value_caps_train_and_valid.pkl"):

        '''
        Load the dataset. (Assumes the current working directory is the project folder.)
        '''

        # Code heavily inspired if not directly from examples.py on project description page.

        #data_path = os.path.join(os.getcwd(), split)
        data_path = "/Users/williamperrault/Desktop/University/Maitrise/1st_year/H2017/IFT6266/Projet/inpainting/train2014"

        #caption_path = os.path.join(os.getcwd(), caption_path)
        caption_path="/Users/williamperrault/Desktop/University/Maitrise/1st_year/H2017/IFT6266/Projet/inpainting/dict_key_imgID_value_caps_train_and_valid.pkl"
        #with open(caption_path) as fd:
        #    caption_dict = pkl.load(fd)

        print (data_path + "/*.jpg")
        imgs = glob.glob(data_path + "/*.jpg")
        Data=np.empty([len(imgs),64,64,3])

        file_list = glob.glob(data_path)

        if "normalized_data.npy" in file_list:
             f = open('normalized_data.npy', 'rb')
             Data = np.load(f)
             f.close()

        else:

            for i, img_path in enumerate(imgs):
                img = Image.open(img_path)
                img_array = np.array(img)

                if img.mode != 'RGB':
                    Data[i,:,:,0]=img_array/255.0
                else:
                    Data[i,:,:,:]=img_array/255.0

                if False:
                    cap_id = os.path.basename(img_path)[:-4]

                    ### Get input/target from the images
                    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
                    if len(img_array.shape) == 3:
                        input = np.copy(img_array)
                        input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                        target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                    else:
                        input = np.copy(img_array)
                        input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                        target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]


                if False:
                    #Image.fromarray(img_array).show()
                    Image.fromarray(input).show()
                    Image.fromarray(target).show()
                    print (i, caption_dict[cap_id])

                if i % 500 == 0:
                    break

        end_time = timeit.default_timer()

        print ("Done loading data. Code ran for : ", (end_time - start_time) / 60. , "minutes")

        return Data # Data_hole (input)

def normalize_input(split="inpainting/train2014"):
    pass
    #if not os.path.isdir(split + "_normalized"):

def train_DCGAN(learning_rate=0.01,n_epochs=10,batch_size=20):

    # Load the dataset
    X_train = load_data()

    # Prepare Theano variables for inputs and targets
    noise = T.tensor4('noise')
    real_image = T.tensor4('real_image')

    # compute number of minibatches for training, validation and testing
    n_train_batches = X_train.shape[0] // batch_size

    # Create neural network model
    print (".........building the model")

    generator = build_generator(noise)
    gen_image = lasagne.layers.get_output(generator)

    discriminator = build_discriminator(gen_image)

    prediction_gen = lasagne.layers.get_output(discriminator)
    prediction_real = lasagne.layers.get_output(discriminator,inputs=real_image)

    # Calculate Loss (BCE):
    loss_gen = lasagne.objectives.binary_crossentropy(prediction_gen,0)
    loss_gen = loss_gen.mean()

    loss_dis = lasagne.objectives.binary_crossentropy(prediction_real,1) + lasagne.objectives.binary_crossentropy(prediction_gen,0)
    loss_dis = loss_dis.mean()

    params_gen = lasagne.layers.get_all_params(generator, trainable=True)
    params_dis = lasagne.layers.get_all_params(discriminator, trainable=True)

    updates_gen = lasagne.updates.adam(loss_gen, params_gen, learning_rate=0.0002, beta1=0.5, beta2=0.999, epsilon=1e-08)
    updates_dis = lasagne.updates.adam(loss_dis, params_dis, learning_rate=0.0002, beta1=0.5, beta2=0.999, epsilon=1e-08)

    # Compile
    train_gen_fn = theano.function([noise], loss_gen,allow_input_downcast=True, updates=updates_gen)
    train_dis_fn = theano.function([noise, real_image], loss_dis,allow_input_downcast=True,on_unused_input='ignore', updates=updates_dis)

    # Training Loop

    print(".........begin Training")

    pdb.set_trace()

    #TO DO: test run une itération avec minibatch de deux et check si output results

    # The following code is heavily based if not directly from the MLP tutorial from http://deeplearning.net/tutorial/mlp.html

    #validation_frequency = min(n_train_batches) #, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_iter = 0
    test_score = 0

    epoch = 0
    done_looping = False

    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            batch = X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            noise = generate_noise(batch_size=batch_size,noise_length=100)
            minibatch_avg_cost_dis = train_dis_fn(noise,batch)
            minibatch_avg_cost_gen = train_gen_fn(noise)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 20 == 0:
                print ("Discriminator Loss : ", loss_dis)
                print ("Generator Loss : " , loss_gen)

        # for i in range (gen_output.shape(0)):
        #     Image.fromarray(image_example[i,:,:,:]).show()
        if epoch % 5:
            generator_params = lasagne.layers.get_all_param_values(generator)
            discriminator_params = lasagne.layers.get_all_param_values(discriminator)

            f = open('gen_params.save', 'wb')
            cPickle.dump(generator_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

            g = open('dis_params.save', 'wb')
            cPickle.dump(discriminator_params, g, protocol=cPickle.HIGHEST_PROTOCOL)
            g.close()




    end_time = timeit.default_timer()

    print ("Code ran for : ", (end_time - start_time) / 60.)

if __name__ == '__main__':
    print ("Running Main")
    start_time = timeit.default_timer()
    train_DCGAN()
