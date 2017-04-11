import lasagne
import numpy as np
import theano
import theano.tensor as T

def build_mlp(input_var=None):
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


# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
image = T.tensor4('image')
real_image = T.tensor4('real_image')

target_var = T.ivector('targets')
# Create neural network model
network = build_mlp(input_var)
image = lasagne.layers.get_output(network)

discriminator = build_discriminator(image)

prediction = lasagne.layers.get_output(discriminator)
prediction_real = lasagne.layers.get_output(discriminator,inputs=real_image)

loss_gen = lasagne.objectives.binary_crossentropy(prediction,0)
loss_dis = lasagne.objectives.binary_crossentropy(prediction_real,1) + lasagne.objectives.binary_crossentropy(prediction,0)

train_gen_fn = theano.function([input_var], loss_gen,allow_input_downcast=True)

train_dis_fn = theano.function([image, real_image], loss_dis,allow_input_downcast=True,on_unused_input='ignore')

input_var =  np.random.normal(loc=0.0, scale=1.0, size=(1,100,1,1))
image = np.random.normal(loc=0.0, scale=1.0, size=(1,3,64,64))
real_image = np.random.normal(loc=0.0, scale=1.0, size=(1,3,64,64))


temp = train_gen_fn(input_var)
temp2 = train_dis_fn(image,real_image)
print temp2.shape
