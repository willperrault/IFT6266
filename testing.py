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

def load_divide_data(split="inpainting/train2014", caption_path="inpainting/dict_key_imgID_value_caps_train_and_valid.pkl"):

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

            if "normalized_data.npy" in glob.glob("/Users/williamperrault/Desktop/University/Maitrise/1st_year/H2017/IFT6266/Projet"):
                 f = open('normalized_data.npy', 'rb')
                 Data = np.load(f)
                 f.close()

            else:

                for i, img_path in enumerate(imgs):
                    img = Image.open(img_path)
                    img_array = np.array(img)

                    if img.mode != 'RGB':
                        continue
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

            end_time = timeit.default_timer()

            print ("Done loading data. Code ran for : ", (end_time - start_time) / 60. , "minutes")

            pdb.set_trace()
            split_data=np.array_split(Data,4)

            for i,data in enumerate(split_data):
                name = 'normalized_data' + str(i)
                np.save(name, data)

            return 'Done splitting data' # Data_hole (input)

start_time = timeit.default_timer()
load_divide_data()

#
#
# import lasagne
# import numpy as np
# import theano
# import theano.tensor as T
#
# def build_mlp(input_var=None):
#     # Input will be noise injected to generate an image
#
#     generator = lasagne.layers.InputLayer(shape=(None,100,1,1),input_var=input_var)
#
#     generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=256,filter_size=(4,4), stride=(1,1)))
#
#     generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=128,filter_size=(5,5), stride=(1,1)))
#
#     generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=64,filter_size=(4,4), stride=(2,2),crop=1))
#
#     generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=32,filter_size=(6,6), stride=(2,2),crop=2))
#
#     generator = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(generator,num_filters=3,filter_size=(6,6), stride=(2,2),crop=2))
#
#     return generator
#
# def build_discriminator(input_var=None):
#
#     discriminator = lasagne.layers.InputLayer(shape=(None,3,64,64),input_var=input_var)
#
#     discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=64,filter_size=(5,5), stride=(2,2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))
#
#     discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=128,filter_size=(5,5), stride=(2,2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))
#
#     discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=256,filter_size=(5,5), stride=(2,2), nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))
#
#     discriminator = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(discriminator,num_filters=100,filter_size=(5,5), stride=(2,2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),pad='same'))
#
#     # Output layer:
#     discriminator = lasagne.layers.DenseLayer(discriminator, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
#
#     return discriminator
#
#
# # Prepare Theano variables for inputs and targets
# input_var = T.tensor4('inputs')
# image = T.tensor4('image')
# real_image = T.tensor4('real_image')
#
# target_var = T.ivector('targets')
# # Create neural network model
# network = build_mlp(input_var)
# image = lasagne.layers.get_output(network)
#
# discriminator = build_discriminator(image)
#
# prediction = lasagne.layers.get_output(discriminator)
# prediction_real = lasagne.layers.get_output(discriminator,inputs=real_image)
#
# loss_gen = lasagne.objectives.binary_crossentropy(prediction,0)
# loss_dis = lasagne.objectives.binary_crossentropy(prediction_real,1) + lasagne.objectives.binary_crossentropy(prediction,0)
#
# train_gen_fn = theano.function([input_var], loss_gen,allow_input_downcast=True)
#
# train_dis_fn = theano.function([image, real_image], loss_dis,allow_input_downcast=True,on_unused_input='ignore')
#
# input_var =  np.random.normal(loc=0.0, scale=1.0, size=(1,100,1,1))
# image = np.random.normal(loc=0.0, scale=1.0, size=(1,3,64,64))
# real_image = np.random.normal(loc=0.0, scale=1.0, size=(1,3,64,64))
#
#
# temp = train_gen_fn(input_var)
# temp2 = train_dis_fn(image,real_image)
# print temp2.shape
