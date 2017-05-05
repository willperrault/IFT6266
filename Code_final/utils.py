import os
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

def load_divide_data(train=True):

            start_time = timeit.default_timer()

            '''
            Load the dataset. (Assumes the current working directory is the project folder.)
            '''

            # Code heavily inspired if not directly from examples.py on project description page.

            #data_path = os.path.join(os.getcwd(), split)
            if train:
                data_path = "/Users/williamperrault/Desktop/University/Maitrise/1st_year/H2017/IFT6266/Projet/inpainting/train2014"
            else:
                data_path = "/Users/williamperrault/Desktop/University/Maitrise/1st_year/H2017/IFT6266/Projet/inpainting/val2014"

            #caption_path = os.path.join(os.getcwd(), caption_path)
            caption_path="/Users/williamperrault/Desktop/University/Maitrise/1st_year/H2017/IFT6266/Projet/inpainting/dict_key_imgID_value_caps_train_and_valid.pkl"
            #with open(caption_path) as fd:
            #    caption_dict = pkl.load(fd)

            print (data_path + "/*.jpg")
            imgs = glob.glob(data_path + "/*.jpg")

            Data=np.empty([len(imgs),64,64,3])

            file_list = glob.glob(data_path)

            for i, img_path in enumerate(imgs):
                img = Image.open(img_path)
                img_array = np.array(img)

                if img.mode != 'RGB':
                    continue
                else:
                    Data[i,:,:,:]=img_array/255.0

                if False:

                    ### Get input/target from the images
                    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
                    if len(img_array.shape) == 3:
                        input = np.copy(img_array)
                        input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                        # target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                    else:
                        input = np.copy(img_array)
                        input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                        target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

                if False:
                    #Image.fromarray(img_array).show()
                    Image.fromarray(input).show()
                    Image.fromarray(target).show()
                    print (i, caption_dict[cap_id])

            end_time = timeit.default_timer()

            print ("Done loading data. Code ran for : ", (end_time - start_time) / 60. , "minutes")

            if train:
                split_data=np.array_split(Data,4)
            else:
                split_data=np.array_split(Data,2)

            for i,data in enumerate(split_data):
                temp=data.astype(dtype='float32')
                temp=np.swapaxes(temp,2,3)
                temp=np.swapaxes(temp,1,2)
                if train:
                    name = 'norm_data' + str(i)
                else:
                    name = 'norm_data_valid' + str(i)
                np.save(name, temp)

            return 'Done splitting data' # Data_hole (input)

def load_data(data_num=0, train=True):
    """ Function to load the either training or validation images from one of the 4 training files or 2 validation files (numbered respectively 0 to 3 or 0 and 1)"""

    if os.getcwd() == '/Users/williamperrault/Github/H2017/IFT6266_final/Code_final':
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

def extract_target(image):

    center = (
        int(np.floor(image.shape[2] / 2.)), int(np.floor(image.shape[3] / 2.)))

    context = np.copy(image)
    context[:, :, center[0] - 16:center[0] +
          16, center[1] - 16:center[1] + 16] = 0

    targets = image[:, :, center[0] - 16:center[0] +
                 16, center[1] - 16:center[1] + 16]

    return context, targets

def show_results(model_name):
    samples = 3
    model_name = model_name + '_'

    # # Create plots of the training values
    # with open (model_name + 'detailed_training_loss.save', 'rb') as fp:
    #     detailed_training_loss = pkl.load(fp)
    #     detailed_training_loss = [float(i) for i in detailed_training_loss]
    #
    # with open (model_name + 'training_loss.save', 'rb') as fp:
    #     training_loss = pkl.load(fp)
    #
    # with open (model_name + 'valid_loss.save', 'rb') as fp:
    #     valid_loss = pkl.load(fp)
    #
    # fg1 = plt.figure()
    # plt.plot(list(range(len(detailed_training_loss))), detailed_training_loss, 'k')
    # plt.xlabel('Minibatch number')
    # plt.ylabel('Loss value')
    # plt.title('Detailed training Loss as a function of every 10 minibatch')
    # fg1.savefig(model_name + 'detailed_training_loss')
    #
    # fg2 = plt.figure()
    # trg = plt.plot(list(range(len(training_loss))), training_loss, 'r',label='Training')
    # vld = plt.plot(list(range(len(training_loss))), valid_loss, 'k',label='Valid')
    # plt.legend()
    # plt.xlabel('Epoch number')
    # plt.ylabel('Loss value')
    # plt.title('Training/Valid loss as a function of epoch')
    # fg2.savefig(model_name + 'valid_vs_training_loss')

    # Generate plots with 3 real images and 3 generated images
    fig = plt.figure()
    #epochs = len(training_loss)

    for epoch in range(33,38):
        generated = np.load(model_name + 'samples' + str(epoch) + '.npy')
        context = np.load(model_name + 'samples_context' + str(epoch) + '.npy' )

        for i in range(3):

            result = np.copy(context[i,:,:,:])
            center = (int(np.floor(result.shape[0] / 2.)), int(np.floor(result.shape[1] / 2.)))

            result[ center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16,:] = generated[i,:,:,:]

            plt.axis('off')
            ax = plt.subplot(2, samples, i + 1)
            ax.imshow(context[i,:,:].astype("uint8"))
            plt.axis('off')
            ax1 = plt.subplot(2, samples, (i + 1) + samples)
            ax1.imshow(result.astype("uint8"))

        fig_name = model_name + 'images' + str(epoch)
        fig.savefig(fig_name)

def temp():

    model_name = 'CG_'
    samples = 3
    fig = plt.figure()
    for epoch in range(1,2):
        generated = np.load(model_name + 'samples' + str(epoch) + '.npy')
        context = np.load(model_name + 'samples_context' + str(epoch) + '.npy' )

        for i in range(3):

            result = np.copy(context[i,:,:,:])
            center = (int(np.floor(result.shape[0] / 2.)), int(np.floor(result.shape[1] / 2.)))

            result[ center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16,:] = generated[i,:,:,:]

            plt.axis('off')
            ax = plt.subplot(2, samples, i + 1)
            ax.imshow(context[i,:,:].astype("uint8"))
            plt.axis('off')
            ax1 = plt.subplot(2, samples, (i + 1) + samples)
            ax1.imshow(result.astype("uint8"))

        fig_name = model_name + 'images' + str(epoch)
        fig.savefig(fig_name)

def temp2():

    with open('CG_detailed_training_loss_gen.save', 'rb') as fp:
        detailed_training_loss1 = pkl.load(fp)
        detailed_training_loss1 = [float(i) for i in detailed_training_loss1]
        print(detailed_training_loss1)

    with open('CG_detailed_training_loss_dis.save', 'rb') as fp:
        detailed_training_loss2 = pkl.load(fp)
        detailed_training_loss2 = [float(i) for i in detailed_training_loss2]
        print(detailed_training_loss2)

    with open('CG_training_loss_gen.save', 'rb') as fp:
        training_loss1 = pkl.load(fp)

    with open('CG_training_loss_dis.save', 'rb') as fp:
        training_loss2 = pkl.load(fp)

def generate_samples_ae():

    sample_context = T.tensor4('target_image',dtype=theano.config.floatX)

    autoencoder = build_autoencoder(input_var=None)

    generated_img = lasagne.layers.get_output(autoencoder, inputs=sample_context)

    output_ae_fn = theano.function(
        [sample_context], generated_img, allow_input_downcast=True)

    # Load original images and their number from the loaded file
    data, n_examples = load_data(0, train=False)

    # Split the image in context and target
    context,targets = extract_target(data)

    fig = plt.figure()
    for i in range(1,51):

        with open('AE_params_best.save'+str(i), 'rb') as fp:
            params = cPickle.load(fp)
            lasagne.layers.set_all_param_values(autoencoder, params)

        rand_int = np.random.randint(0, high=n_examples-3)
        real_images = np.copy(data[rand_int:rand_int+3,:,:,:]*255.0)
        sample_context = np.copy(context[rand_int:rand_int+3,:,:,:])

        generated = output_ae_fn(sample_context) * 255.0
        generated = np.swapaxes(generated, 1, 2)
        generated = np.swapaxes(generated, 2, 3)

        real_images = np.swapaxes(real_images, 1, 2)
        real_images = np.swapaxes(real_images, 2, 3)

        samples = 3
        for j in range(3):

            result = np.copy(real_images[j,:,:,:])
            center = (int(np.floor(result.shape[0] / 2.)), int(np.floor(result.shape[1] / 2.)))

            result[ center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16,:] = generated[j,:,:,:]

            plt.axis('off')
            ax = plt.subplot(2, samples, j + 1)
            ax.imshow(real_images[j,:,:].astype("uint8"))
            plt.axis('off')
            ax1 = plt.subplot(2, samples, (j + 1) + samples)
            ax1.imshow(result.astype("uint8"))

        fig_name = 'AE_images_train' + str(i)
        fig.savefig(fig_name)
    plt.close()
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


if __name__ == '__main__':
    # generate_samples_ae()
    show_results('CG')
