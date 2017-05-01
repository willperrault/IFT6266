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
