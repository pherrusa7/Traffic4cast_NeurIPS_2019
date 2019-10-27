import numpy as np
import h5py
import os, csv, re, datetime
import sys, getopt
import importlib

import tensorflow as tf
sess = tf.compat.v1.Session()

from keras import backend as K
K.set_session(sess)

from keras import models, activations
from keras import layers
from keras import optimizers
from keras import losses
# from keras import backend as K 
from tensorflow import device

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical  

# imports for RAE_w_SC
from keras import layers
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, ZeroPadding2D, Cropping2D, TimeDistributed
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras import losses, optimizers

# own libraries
from src.losses import *

######################################################################
###################################################### SIMPLE CONVLSTM
######################################################################

def build_model():
    """Build keras model.
    
        Returns: keras model
    """
    
    #define model input
    data_shape = (3, 3, 495, 436)
    prev_frames = layers.Input(shape=data_shape, name='prev_frames')
    
    #define layers
    with device('/device:GPU:0'):
        convlstm_0 = layers.ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True, return_state=False,
                             activation='tanh', recurrent_activation='hard_sigmoid',
                             kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                             data_format='channels_first', name='convlstm_0')
        
        convlstm_1 = layers.ConvLSTM2D(filters=64, kernel_size=(7, 7), padding='same', return_sequences=False, return_state=True,
                             activation='tanh', recurrent_activation='hard_sigmoid',
                             kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                             data_format='channels_first', name='convlstm_1')
    
    with device('/device:GPU:1'):
        convlstm_2 = layers.ConvLSTM2D(filters=64, kernel_size=(7, 7), padding='same', return_sequences=True, return_state=False,
                             activation='tanh', recurrent_activation='hard_sigmoid',
                             kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                             data_format='channels_first', name='convlstm_2')
        
        convlstm_3 = layers.ConvLSTM2D(filters=3, kernel_size=(7, 7), padding='same', return_sequences=True, return_state=False,
                             activation='relu', recurrent_activation='hard_sigmoid',
                             kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                             data_format='channels_first', name='convlstm_3')
    
        
    #define model structure  
    #encoder      
    x = convlstm_0(prev_frames)
    x = convlstm_1(x)[-1]
    
    #flatten, repeat and reshape
    x = layers.Flatten()(x)
    x = layers.RepeatVector(3)(x)
    x = layers.Reshape((3, 64, 495, 436))(x)
    
    #decoder
    x = convlstm_2(x)
    x = convlstm_3(x)
            
    #build and return model
    seq_model = models.Model(inputs=prev_frames, outputs=x)
    return seq_model

############################################################################################
###################################################### CONVLSTM + CLASSIFICATION FOR HEADING
############################################################################################

def add_clf_layer(model, num_out=5, categorical_axis=2):
    """ take an existing model and add a second output for classification of 'num_out' classes """
    
    with device('/device:GPU:0'):
            convlstm_clf = layers.ConvLSTM2D(filters=num_out, kernel_size=(7, 7), padding='same', return_sequences=True, return_state=False,
                                 activation='relu', recurrent_activation='hard_sigmoid',
                                 kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                                 data_format='channels_first', name='convlstm_clf')

    # add the layer from the ConvLstm layer previous to the output
    x = convlstm_clf(model.layers[-2].output)
    x = layers.Softmax(axis=categorical_axis, name='softmax_clf')(x)
    
    # reshape so we get a vector of probabilities for each pixel
    #x = layers.Reshape((-1, num_out))(x)

#     model2 = models.Model(input=model.input, output=[model.layers[-1].output, x]) # old keras version
    model2 = models.Model(inputs=[model.input], outputs=[model.layers[-1].output, x])
    
    return model2

def get_2losses_model(model_path, model_name, weights_initial_model=None, weights_current_model=None, add_model=add_clf_layer, add_loss=cross_entropy_with_axis(), sample_weight_mode=None):    

    # load the old model and its weights
    model = get_model(model_path, model_name, weights_initial_model)

    # Add another layer for classification
    model = add_model(model, num_out=5)
    
    if weights_current_model is not None:
        print('loading weitghs new:', weights_current_model)
        model.load_weights(weights_current_model)
    
    print(sample_weight_mode)
    # compile with a custom loss that handle both regression and classification
    lr = 0.0001
    optimizer = optimizers.Adam(lr=lr)
    print("lr:", lr)
    model.compile(optimizer=optimizer,
                  loss={'convlstm_3': losses.mean_squared_error, 'softmax_clf': add_loss},
                  loss_weights={'convlstm_3': 1., 'softmax_clf': 0.0105}, 
                  sample_weight_mode=sample_weight_mode)
#     model.compile(optimizer=optimizer, loss=[losses.mean_squared_error, add_loss], sample_weight_mode=sample_weight_mode)
    
    return model

#################################################################################################################################
###################################################### CONVLSTM + CLASSIFICATION FOR HEADING WITH SPARSE CrossEntropy AND WEIGHTS
#################################################################################################################################

def add_clf_layer_sparse(model, num_out=5, categorical_axis=2):
    """ take an existing model and add a second output for classification of 'num_out' classes """
    
    with device('/device:GPU:0'):
            convlstm_clf = layers.ConvLSTM2D(filters=num_out, kernel_size=(7, 7), padding='same', return_sequences=True, return_state=False,
                                 activation='tanh', recurrent_activation='hard_sigmoid',
                                 kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                                 data_format='channels_first', name='convlstm_clf')

    # add the layer from the ConvLstm layer previous to the output
    x = convlstm_clf(model.layers[-2].output)
    
    # move dimension 'heading' to predict at the end, so softmax can be compute for each pixel for each frame
    x = layers.Permute((1, 3, 4, 2))(x)  # this way we control reshape
    x = layers.Reshape((-1, num_out))(x) # vectorize all frames

    x = layers.Softmax(name='softmax_clf')(x)

    #build and return model
#     seq_model = models.Model(input=model.input, output=[model.layers[-1].output, x]) # old keras version
    model2 = models.Model(inputs=[model.input], outputs=[model.layers[-1].output, x])
    
    return model2

##################################################################################################
###################################################### Recursive Autoencoder with skip connections
##################################################################################################
# with relu before BN
# def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, name='', batchnorm_implement=True):
#     # first layer
#     x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
#                padding="same", name=name+'_conv1', activation='relu')(input_tensor)
#     if batchnorm_implement:
#         x = BatchNormalization(trainable=batchnorm)(x)
# #     x = Activation("relu")(x)
#     # second layer
#     x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
#                padding="same", name=name+'_conv2', activation='relu')(x)
#     if batchnorm_implement:
#         x = BatchNormalization(trainable=batchnorm)(x)
# #     x = Activation("relu")(x)
#     return x


# # try depth conv
# def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, name='', batchnorm_implement=True):
#     # first layer
#     x = layers.SeparableConv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same", name=name+'_conv1')(input_tensor)
#     if batchnorm_implement:
#         x = BatchNormalization(trainable=batchnorm)(x)
#     x = Activation("relu")(x)
#     # second layer
#     x = layers.SeparableConv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same", name=name+'_conv2')(x)
#     if batchnorm_implement:
#         x = BatchNormalization(trainable=batchnorm)(x)
#     x = Activation("relu")(x)
#     return x

# regular
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, name='', batchnorm_implement=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same", name=name+'_conv1')(input_tensor)
    if batchnorm_implement:
        x = BatchNormalization(trainable=batchnorm)(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same", name=name+'_conv2')(x)
    if batchnorm_implement:
        x = BatchNormalization(trainable=batchnorm)(x)
    x = Activation("relu")(x)
    return x

def get_encoder(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    """ modified from https://www.depends-on-the-definition.com/unet-keras-segmenting-images/ """
    
    # padding with zeros to make input 512x512 
    c1 = ZeroPadding2D(padding=((8,9), (38,38)))(input_img)

    # contracting path
    c1 = conv2d_block(c1, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, name='c1')
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, name='c2')
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, name='c3')
    c3_1 = conv2d_block(c3, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, name='c3_1')
    p3 = MaxPooling2D((2, 2)) (c3_1)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, name='c4')
    c4_1 = conv2d_block(c4, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, name='c4_1')
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4_1)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, name='c5')
    c5_1 = conv2d_block(c5, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, name='c6_1')
    c5_1 = MaxPooling2D(pool_size=(2, 2)) (c5_1)
    
    c5_1 = conv2d_block(c5_1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, name='c6_2')
    c5_1 = MaxPooling2D(pool_size=(2, 2)) (c5_1)
    c5_1 = Dropout(dropout)(c5_1)
    
    c5_1 = layers.Reshape(target_shape=(-1,), name='last')(c5_1)
    
    return Model(inputs=[input_img], outputs=[c5_1, c5, c4, c3, c2, c1], name='encoder')#

def get_decoder(embedding, skip_connections, n_filters=16, dropout=0.5, batchnorm=True, channels_out=3):
    # load skip connections with shape: (32, 128), (64, 128), (128, 64), (256, 32), (512, 16))
    c4, c3, c2, c1, c0 = skip_connections[0], skip_connections[1], skip_connections[2], skip_connections[3], skip_connections[-1]
    
    # reshape to image  
    x = layers.Reshape((8, 8, 32))(embedding)
    
    x = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (x) # 16
    x = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (x) # 32
    u6 = concatenate([x, c4])
    
    # expansive path:
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same', name='u6') (x) # 64
    u6 = concatenate([u6, c3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, name='c6')

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same', name='u7') (c6) # 128
    u7 = concatenate([u7, c2])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, name='c7')

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same', name='u8') (c7) # 256
    u8 = concatenate([u8, c1])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, name='c8')

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same', name='u9') (c8) # 512
    u9 = concatenate([u9, c0])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, name='c9')
    
    # crop to original size
    c9 = Cropping2D(cropping=((8,9), (38,38)))(c9)
    outputs = conv2d_block(c9, n_filters=channels_out, kernel_size=3, batchnorm=batchnorm, name='out')

    
    return Model(inputs=[embedding, c4, c3, c2, c1, c0], outputs=[outputs], name='decoder')

def ae_model():
    enc, dec = get_encoder_decoder()
    im_height=495
    im_width=436
    channels=3

    input_img = Input((im_height, im_width, channels), name='img')

    h, c4, c3, c2, c1, c0 = enc(input_img)
    print(h.shape, c4.shape)
    h = dec([h, c4, c3, c2, c1, c0])
    print(h.shape)

    #     return Model(inputs=[input_img], outputs=[x], name='autoencoder')
    ae = Model(inputs=[input_img], outputs=[h], name='autoencoder')

    return ae   

def get_encoder_decoder(embedding_size=2048, im_height=495, im_width=436, channels=3, 
                        dropout_enc=0.05, dropout_dec=0.05, b_norm_enc=True, b_norm_dec=True):

    # define inputs
    img_input_shape = (im_height, im_width, channels)
    input_img = Input(img_input_shape, name='img')
    input_emb = layers.Input((embedding_size,), name='embedding')
    
    skip_0 = Input((32, 32, 128), name='skip_0')
    skip_1 = Input((64, 64, 128), name='skip_1')
    skip_2 = Input((128, 128, 64), name='skip_2')
    skip_3 = Input((256, 256, 32), name='skip_3')
    skip_4 = Input((512, 512, 16), name='skip_4')

    skip_connections_inputs = [skip_0, skip_1, skip_2, skip_3, skip_4]
    
    # get encoder/decoder models
    with tf.name_scope('encoder'):
        encoder = get_encoder(input_img, n_filters=16, dropout=dropout_enc, batchnorm=b_norm_enc)
    with tf.name_scope('decoder'):
        decoder = get_decoder(input_emb, skip_connections_inputs, n_filters=16, dropout=dropout_dec, batchnorm=b_norm_dec)
    

    return encoder, decoder



def get_RAEwSC(params, embedding_size=2048, im_height=495, im_width=436, channels=3, length_seq_in=3, length_seq_out=3, 
               dropout_enc=0.05, dropout_dec=0.05, b_norm_enc=True, b_norm_dec=True):
    """
    Recurrent Autoencoder with Skip Connections (RAEwSC)
    multiple outputs & TimeDistributed: https://github.com/keras-team/keras/issues/6449
    multiple inputs & TimeDistributed:
    """
    
    # define inputs
    img_input_shape = (im_height, im_width, channels)
    seq_input_shape = (length_seq_in,) + img_input_shape
    prev_frames = layers.Input(shape=seq_input_shape, name='prev_frames')
    seq_output_shape = (length_seq_out,) + img_input_shape
    future_frames = layers.Input(shape=seq_output_shape, name='future_frames')

    # define encoder/decoder models for a single frame
    encoder, decoder = get_encoder_decoder(embedding_size, im_height, im_width, channels,
                                           dropout_enc=dropout_enc, dropout_dec=dropout_dec, 
                                           b_norm_enc=b_norm_enc, b_norm_dec=b_norm_dec)
    
    #################################################### Encoder Phase
    ####### PAST
    # get embeddings for each frame in the sequence
    embeddings = []
    for i in range(length_seq_in):
        # slice & encoder for current frame
        current_frame = Lambda(lambda x: x[:, i], output_shape=(im_height, im_width, channels))(prev_frames)
        h, c4, c3, c2, c1, c0 = encoder(current_frame)

        # append encoders adding the sequence dimension to later concatenate them
        h = Reshape( (1, embedding_size) )(h)
        embeddings.append(h)
    embeddings = concatenate(embeddings, axis=1, name='Concat_embeddings')
    print("embeddings.shape:", embeddings.shape)
    
    ####### FUTURE
    # encode future frames to guide the recurrent-manifold construction
    future_embeddings = []
    for i in range(length_seq_out):
        # slice & encoder for current frame
        current_frame = Lambda(lambda x: x[:, i], output_shape=(im_height, im_width, channels))(future_frames)
        h, _, _, _, _, _ = encoder(current_frame)

        # append encoders adding the sequence dimension to later concatenate them
        h = Reshape( (1, embedding_size) )(h)
        future_embeddings.append(h)
    future_embeddings = concatenate(future_embeddings, axis=1, name='Concat_future_emb')
    print("future_embeddings.shape:", future_embeddings.shape)
    
    #################################################### Recurrent Phase
    # time encoder
    embeddings = layers.GRU(params['gru_enc_1'], return_sequences = True, name='gru_enc_1')(embeddings)
    embeddings = layers.GRU(params['gru_enc_2'], return_sequences = False, name='gru_enc_2')(embeddings)

    embeddings = RepeatVector(length_seq_out, name='repeat_vector')(embeddings)

    # time decoder
    embeddings = layers.GRU(params['gru_dec_1'], return_sequences = True, name='gru_dec_1')(embeddings)
    embeddings = layers.GRU(params['gru_dec_2'], return_sequences = True, name='gru_dec_2')(embeddings)
    embeddings = layers.GRU(embedding_size, return_sequences = True, name='gru_dec_3')(embeddings)
    print("recurrent embeddings.shape:", embeddings.shape)
    
    #################################################### Decoder Phase
    # get decoder for each frame predicted in the sequence (using skip connection from the most known recent frame)
    prediced_frames = []
    for i in range(length_seq_out):
        # slice & decoder for current frame
        current_embedding = Lambda(lambda x: x[:, i], output_shape=(embedding_size,))(embeddings)
        current_frame = decoder([current_embedding, c4, c3, c2, c1, c0])

        # append frames adding the sequence dimension to later concatenate them
        current_frame = Reshape( (1,)+img_input_shape )(current_frame)
        prediced_frames.append(current_frame)
    prediced_frames = concatenate(prediced_frames, axis=1, name='Concat_predicted_frames')
    print("prediced_frames.shape:", prediced_frames.shape)
    
    return Model(inputs=[prev_frames, future_frames], outputs=[prediced_frames], name='RAE_w_SC'), embeddings, future_embeddings

def get_RAEwSC_compiled(weights=None, lr = 0.001, grad_clip=1., loss_weights={'predicted_frames':1., 'predicted_emb':100.}):
    """ define and compile the model """
    
    params = {'gru_enc_1': 256, 'gru_enc_2': 128, 'gru_dec_1': 128, 'gru_dec_2': 256}
    model, embeddings, future_embeddings = get_RAEwSC(params)
    
    if weights is not None:
        print('loading weitghs:', weights)
        model.load_weights(weights)
    
    # compile with a custom loss that handle both regression and classification
    optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) #optimizers.Adam(lr=lr, clipvalue=grad_clip)
    print("lr:", lr)
    
    model.compile(optimizer=optimizer,
                  loss=loss_with_latent_term_2(future_embeddings, embeddings, loss_weights),
                  metrics=[losses.mean_squared_error, get_recurrent_embedding_loss(future_embeddings, embeddings, loss_weights)])
    
    return model

#############################################################################################################################################
###################################################### Recursive Autoencoder with skip connections & Exogenous Variables (time-flow, weather)
#############################################################################################################################################

def get_RAEwSC_and_ExogenousVars(params, embedding_size=2048, im_height=495, im_width=436, channels=3, length_seq_in=3, length_seq_out=3, 
                                 dropout_enc=0.05, dropout_dec=0.05, b_norm_enc=True, b_norm_dec=True):
    """
    Recurrent Autoencoder with Skip Connections (RAEwSC)
    multiple outputs & TimeDistributed: https://github.com/keras-team/keras/issues/6449
    multiple inputs & TimeDistributed:
    """
    
    # define inputs
    img_input_shape = (im_height, im_width, channels)
    seq_input_shape = (length_seq_in,) + img_input_shape
    prev_frames = layers.Input(shape=seq_input_shape, name='prev_frames')
    seq_output_shape = (length_seq_out,) + img_input_shape
    future_frames = layers.Input(shape=seq_output_shape, name='future_frames')

    # define encoder/decoder models for a single frame
    encoder, decoder = get_encoder_decoder(embedding_size, im_height, im_width, channels,
                                           dropout_enc=dropout_enc, dropout_dec=dropout_dec, 
                                           b_norm_enc=b_norm_enc, b_norm_dec=b_norm_dec)
    
    #################################################### Encoder Phase
    ####### PAST
    # get embeddings for each frame in the sequence
    embeddings = []
    for i in range(length_seq_in):
        # slice & encoder for current frame
        current_frame = Lambda(lambda x: x[:, i], output_shape=(im_height, im_width, channels))(prev_frames)
        h, c4, c3, c2, c1, c0 = encoder(current_frame)

        # append encoders adding the sequence dimension to later concatenate them
        h = Reshape( (1, embedding_size) )(h)
        embeddings.append(h)
    embeddings = concatenate(embeddings, axis=1, name='Concat_embeddings')
    print("embeddings.shape:", embeddings.shape)
    
    ####### FUTURE
    # encode future frames to guide the recurrent-manifold construction
    future_embeddings = []
    for i in range(length_seq_out):
        # slice & encoder for current frame
        current_frame = Lambda(lambda x: x[:, i], output_shape=(im_height, im_width, channels))(future_frames)
        h, _, _, _, _, _ = encoder(current_frame)

        # append encoders adding the sequence dimension to later concatenate them
        h = Reshape( (1, embedding_size) )(h)
        future_embeddings.append(h)
    future_embeddings = concatenate(future_embeddings, axis=1, name='Concat_future_emb')
    print("future_embeddings.shape:", future_embeddings.shape)
    
    #################################################### Add Exogenous Vars Phase: day_input, time_input
    # time-flow inputs
    day_input_shape = (length_seq_in, (1+length_seq_out)*50)
    time_input_shape = (length_seq_in, (1+length_seq_out)*2)
    day_input = layers.Input(shape=day_input_shape, name='day_info')
    time_input = layers.Input(shape=time_input_shape, name='time_info')
    
    # weather inputs 
    weather_categorical_input_shape = (length_seq_in, (1+length_seq_out)*28)
    weather_continous_input_shape = (length_seq_in, (1+length_seq_out)*5)
    weather_categorical_input = layers.Input(shape=weather_categorical_input_shape, name='weather_categorical')
    weather_continous_input = layers.Input(shape=weather_continous_input_shape, name='weather_continous')
    
    # embedding for categorical data
#     weather_categorical_input = layers.TimeDistributed( layers.Dense(params['embed_weather'], name='embed_weather') )(weather_categorical_input)
    
    # concat visual and exogenous varibales & combine with a FC layer
    embeddings = concatenate([embeddings, day_input, time_input, weather_categorical_input, weather_continous_input], axis=-1, name='Concat_exogenous')
    print("concatenation of all inputs:", embeddings.shape)
    embeddings = layers.TimeDistributed( layers.Dense(params['units_before_recurrent'], name='embedding_FC') )(embeddings)
    print("FC before recurrent embeddings.shape:", embeddings.shape)
    
    #################################################### Recurrent Phase
    # time encoder
    embeddings = layers.GRU(params['gru_enc_1'], return_sequences = True, name='gru_enc_1_FC')(embeddings)
    embeddings = layers.GRU(params['gru_enc_2'], return_sequences = False, name='gru_enc_2')(embeddings)

    embeddings = RepeatVector(length_seq_out, name='repeat_vector')(embeddings)

    # time decoder
    embeddings = layers.GRU(params['gru_dec_1'], return_sequences = True, name='gru_dec_1')(embeddings)
    embeddings = layers.GRU(params['gru_dec_2'], return_sequences = True, name='gru_dec_2')(embeddings)
    embeddings = layers.GRU(embedding_size, return_sequences = True, name='gru_dec_3')(embeddings)
    print("recurrent embeddings.shape:", embeddings.shape)
    
    #################################################### Decoder Phase
    # get decoder for each frame predicted in the sequence (using skip connection from the most known recent frame)
    prediced_frames = []
    for i in range(length_seq_out):
        # slice & decoder for current frame
        current_embedding = Lambda(lambda x: x[:, i], output_shape=(embedding_size,))(embeddings)
        current_frame = decoder([current_embedding, c4, c3, c2, c1, c0])

        # append frames adding the sequence dimension to later concatenate them
        current_frame = Reshape( (1,)+img_input_shape )(current_frame)
        prediced_frames.append(current_frame)
    prediced_frames = concatenate(prediced_frames, axis=1, name='Concat_predicted_frames')
    print("prediced_frames.shape:", prediced_frames.shape)
    
    return Model(inputs=[prev_frames, future_frames, day_input, time_input, weather_categorical_input, weather_continous_input], 
                 outputs=[prediced_frames], name='RAE_w_SC'), embeddings, future_embeddings

def get_RAEwSC_and_WS_compiled(weights=None, lr = 0.001, grad_clip=1., loss_weights={'predicted_frames':1., 'predicted_emb':1.}, 
                                 dropout_enc=0.05, dropout_dec=0.05, b_norm_enc=True, b_norm_dec=True):
    """ define and compile the model """
    
    params = {'units_before_recurrent': 2048, 'gru_enc_1': 256, 'gru_enc_2': 128, 'gru_dec_1': 128, 'gru_dec_2': 256, "embed_weather": 8}
    model, embeddings, future_embeddings = get_RAEwSC_and_ExogenousVars(params, 
                                           dropout_enc=dropout_enc, dropout_dec=dropout_dec, 
                                           b_norm_enc=b_norm_enc, b_norm_dec=b_norm_dec)
    
    if weights is not None:
        print('loading weitghs:', weights)
        model.load_weights(weights, by_name=True)
    
    # compile with a custom loss that handle both regression and classification
    optimizer = [optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), optimizers.Adam(lr=lr, clipvalue=grad_clip)][1]
    print("lr:", lr, 'optimizer:', optimizer)
    
    model.compile(optimizer=optimizer,
                  loss=loss_with_latent_term_2(future_embeddings, embeddings, loss_weights),
                  metrics=[losses.mean_squared_error, get_recurrent_embedding_loss(future_embeddings, embeddings, loss_weights)])
    
    return model

#############################################################################################################################################
###################################################### Recursive Autoencoder with skip connections & Exogenous Variables (time-flow, weather) + input in Decoder
#############################################################################################################################################

def get_decoder2(embedding, skip_connections, n_filters=16, dropout=0.5, batchnorm=True, channels_out=3):
    # load skip connections with shape: (32, 128), (64, 128), (128, 64), (256, 32), (512, 16))
    c4, c3, c2, c1, c0, in_enc = skip_connections[0], skip_connections[1], skip_connections[2], skip_connections[3], skip_connections[4], skip_connections[-1]
    
    # reshape to image  
    x = layers.Reshape((8, 8, 32))(embedding)
    
    x = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (x) # 16
#     x = conv2d_block(x, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, name='new1') # new
#     x = Dropout(dropout)(x) # new
    x = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (x) # 32
#     x = conv2d_block(x, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, name='new2') # new
#     x = Dropout(dropout)(x) # new
    u6 = concatenate([x, c4])
    
    # expansive path:
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same', name='u6') (x) # 64
    u6 = concatenate([u6, c3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, name='c6')

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same', name='u7') (c6) # 128
    u7 = concatenate([u7, c2])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, name='c7')

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same', name='u8') (c7) # 256
    u8 = concatenate([u8, c1])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, name='c8')

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same', name='u9') (c8) # 512
    u9 = concatenate([u9, c0])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, name='c9')
    
    # crop to original size
    c9 = Cropping2D(cropping=((8,9), (38,38)))(c9)
    c9 = concatenate([c9, in_enc]) # new concat with input
    
#     outputs = conv2d_block(c9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, name='out_dec')
#     outputs = conv2d_block(outputs, n_filters=channels_out, kernel_size=3, batchnorm=batchnorm, name='out_dec2') # new, but n_filters=channels_out
    outputs = conv2d_block(c9, n_filters=channels_out, kernel_size=3, batchnorm=batchnorm, name='out_dec')

    
    return Model(inputs=[embedding, c4, c3, c2, c1, c0, in_enc], outputs=[outputs], name='decoder')

def get_encoder_decoder_w_input(embedding_size=2048, im_height=495, im_width=436, channels=3, 
                        dropout_enc=0.05, dropout_dec=0.05, b_norm_enc=True, b_norm_dec=True):

    # define inputs
    img_input_shape = (im_height, im_width, channels)
    input_img = Input(img_input_shape, name='img')
    input_emb = layers.Input((embedding_size,), name='embedding')
    
    skip_0 = Input((32, 32, 128), name='skip_0')
    skip_1 = Input((64, 64, 128), name='skip_1')
    skip_2 = Input((128, 128, 64), name='skip_2')
    skip_3 = Input((256, 256, 32), name='skip_3')
    skip_4 = Input((512, 512, 16), name='skip_4')

    skip_connections_inputs = [skip_0, skip_1, skip_2, skip_3, skip_4, input_img]
    
    # get encoder/decoder models
    with tf.name_scope('encoder'):
        encoder = get_encoder(input_img, n_filters=16, dropout=dropout_enc, batchnorm=b_norm_enc)
    with tf.name_scope('decoder'):
        decoder = get_decoder2(input_emb, skip_connections_inputs, n_filters=16, dropout=dropout_dec, batchnorm=b_norm_dec)
    

    return encoder, decoder


def get_RAEwSC_and_ExogenousVars_w_input(params, embedding_size=2048, im_height=495, im_width=436, channels=3, length_seq_in=3, length_seq_out=3, 
                                 dropout_enc=0.05, dropout_dec=0.05, b_norm_enc=True, b_norm_dec=True):
    """
    Recurrent Autoencoder with Skip Connections (RAEwSC)
    multiple outputs & TimeDistributed: https://github.com/keras-team/keras/issues/6449
    multiple inputs & TimeDistributed:
    """
    
    # define inputs
    img_input_shape = (im_height, im_width, channels)
    seq_input_shape = (length_seq_in,) + img_input_shape
    prev_frames = layers.Input(shape=seq_input_shape, name='prev_frames')
    seq_output_shape = (length_seq_out,) + img_input_shape
    future_frames = layers.Input(shape=seq_output_shape, name='future_frames')

    # define encoder/decoder models for a single frame
    encoder, decoder = get_encoder_decoder_w_input(embedding_size, im_height, im_width, channels,
                                           dropout_enc=dropout_enc, dropout_dec=dropout_dec, 
                                           b_norm_enc=b_norm_enc, b_norm_dec=b_norm_dec)
    
    #################################################### Encoder Phase
    ####### PAST
    # get embeddings for each frame in the sequence
    embeddings = []
    for i in range(length_seq_in):
        # slice & encoder for current frame
        current_frame = Lambda(lambda x: x[:, i], output_shape=(im_height, im_width, channels))(prev_frames)
        h, c4, c3, c2, c1, c0 = encoder(current_frame)

        # append encoders adding the sequence dimension to later concatenate them
        h = Reshape( (1, embedding_size) )(h)
        embeddings.append(h)
        
    embeddings = concatenate(embeddings, axis=1, name='Concat_embeddings')
    print("embeddings.shape:", embeddings.shape)
    
    ####### FUTURE
    # encode future frames to guide the recurrent-manifold construction
    future_embeddings = []
    for i in range(length_seq_out):
        # slice & encoder for current frame
        current_fut_frame = Lambda(lambda x: x[:, i], output_shape=(im_height, im_width, channels))(future_frames)
        h, _, _, _, _, _ = encoder(current_fut_frame)

        # append encoders adding the sequence dimension to later concatenate them
        h = Reshape( (1, embedding_size) )(h)
        future_embeddings.append(h)
    future_embeddings = concatenate(future_embeddings, axis=1, name='Concat_future_emb')
    print("future_embeddings.shape:", future_embeddings.shape)
    
    #################################################### Add Exogenous Vars Phase: day_input, time_input
    # time-flow inputs
    day_input_shape = (length_seq_in, (1+length_seq_out)*50)
    time_input_shape = (length_seq_in, (1+length_seq_out)*2)
    day_input = layers.Input(shape=day_input_shape, name='day_info')
    time_input = layers.Input(shape=time_input_shape, name='time_info')
    
    # weather inputs 
    weather_categorical_input_shape = (length_seq_in, (1+length_seq_out)*28)
    weather_continous_input_shape = (length_seq_in, (1+length_seq_out)*5)
    weather_categorical_input = layers.Input(shape=weather_categorical_input_shape, name='weather_categorical')
    weather_continous_input = layers.Input(shape=weather_continous_input_shape, name='weather_continous')
    
    # embedding for categorical data
#     weather_categorical_input = layers.TimeDistributed( layers.Dense(params['embed_weather'], name='embed_weather') )(weather_categorical_input)
    
    # concat visual and exogenous varibales & combine with a FC layer
    embeddings = concatenate([embeddings, day_input, time_input, weather_categorical_input, weather_continous_input], axis=-1, name='Concat_exogenous')
    print("concatenation of all inputs:", embeddings.shape)
    embeddings = layers.TimeDistributed( layers.Dense(params['units_before_recurrent'], name='embedding_FC') )(embeddings)
    print("FC before recurrent embeddings.shape:", embeddings.shape)
    
    #################################################### Recurrent Phase
    # time encoder
    embeddings = layers.GRU(params['gru_enc_1'], return_sequences = True, name='gru_enc_1_FC')(embeddings)
    embeddings = layers.GRU(params['gru_enc_2'], return_sequences = False, name='gru_enc_2')(embeddings)

    embeddings = RepeatVector(length_seq_out, name='repeat_vector')(embeddings)

    # time decoder
    embeddings = layers.GRU(params['gru_dec_1'], return_sequences = True, name='gru_dec_1')(embeddings)
    embeddings = layers.GRU(params['gru_dec_2'], return_sequences = True, name='gru_dec_2')(embeddings)
    embeddings = layers.GRU(embedding_size, return_sequences = True, name='gru_dec_3')(embeddings)
    print("recurrent embeddings.shape:", embeddings.shape)
    
    #################################################### Decoder Phase
    # get decoder for each frame predicted in the sequence (using skip connection from the most known recent frame and its inmput)
    prediced_frames = []
    for i in range(length_seq_out):
        # slice & decoder for current frame
        current_embedding = Lambda(lambda x: x[:, i], output_shape=(embedding_size,))(embeddings)
        current_pred_frame = decoder([current_embedding, c4, c3, c2, c1, c0, current_frame])

        # append frames adding the sequence dimension to later concatenate them
        current_pred_frame = Reshape( (1,)+img_input_shape )(current_pred_frame)
        prediced_frames.append(current_pred_frame)
    prediced_frames = concatenate(prediced_frames, axis=1, name='Concat_predicted_frames')
    print("prediced_frames.shape:", prediced_frames.shape)
    
    return Model(inputs=[prev_frames, future_frames, day_input, time_input, weather_categorical_input, weather_continous_input], 
                 outputs=[prediced_frames], name='RAE_w_SC_WT_I'), embeddings, future_embeddings

def get_RAEwSC_and_WS_compiled_w_input(weights=None, length_seq_in=3, lr = 0.001, grad_clip=1., loss_weights={'predicted_frames':1., 'predicted_emb':1.}, 
                                 dropout_enc=0.05, dropout_dec=0.05, b_norm_enc=True, b_norm_dec=True):
    """ define and compile the model """
    
    params = {'units_before_recurrent': 2048, 'gru_enc_1': 256, 'gru_enc_2': 128, 'gru_dec_1': 128, 'gru_dec_2': 256, "embed_weather": 8}
    model, embeddings, future_embeddings = get_RAEwSC_and_ExogenousVars_w_input(params, length_seq_in=length_seq_in, 
                                           dropout_enc=dropout_enc, dropout_dec=dropout_dec, 
                                           b_norm_enc=b_norm_enc, b_norm_dec=b_norm_dec)
    
    if weights is not None:
        print('loading weitghs:', weights)
        model.load_weights(weights, by_name=True)
    
    # compile with a custom loss that handle both regression and classification
    optimizer = [optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), optimizers.Adam(lr=lr, clipvalue=grad_clip)][1]
    print("lr:", lr, 'optimizer:', optimizer)
    
    model.compile(optimizer=optimizer,
                  loss=loss_with_latent_term_2(future_embeddings, embeddings, loss_weights),
                  metrics=[losses.mean_squared_error, get_recurrent_embedding_loss(future_embeddings, embeddings, loss_weights)])
    
    return model

###################################################### LOAD A MODEL FROM ANY FILE BY ITS NAME

def get_model(model_path, function_name, weights=None, opt=None, loss=None):
    """ Load any model and its weights specifying the path of the file containing 
        the model and the name of the function where it is defined
        TODO: accept also any number of parameters
    """
    # get path and file names
    model_path = model_path.split('/')
    path_m, name_m = '/'.join(model_path[:-1]), model_path[-1][:-3] 
    
    # add path to the system
    sys.path.insert(0, path_m)
    
    # import the file where the model is defined
    mod = importlib.import_module(name_m)
    
    # construct the model & load weights
    func = getattr(mod, function_name)
    model = func()
    
    if weights is not None:
        print('loading weitghs:', weights)
        model.load_weights(weights)
        
    if (opt is not None) and (loss is not None):
        model.compile(optimizer=opt, loss=loss)
        
    return model

###################################################### MODEL TO TEST STUFF

def get_identity_model():
    samples_in_sequence, num_out = 3, 5
    data_shape = (samples_in_sequence, num_out, 495, 436) # (3, 5, 495, 436)

    prev_frames = layers.Input(shape=data_shape, name='prev_frames')

    # Literally, the identity function here
    x = layers.Reshape((3, num_out, 495, 436))(prev_frames)

    # move dimension 'heading' to predict at the end, so softmax can be compute for each pixel for each frame
    x = layers.Permute((1, 3, 4, 2))(x)  # this way we control reshape
    x = layers.Reshape((-1, num_out))(x) # vectorize all frames

    x = layers.Softmax(name='softmax_clf')(x)

    #build and return model
    seq_model = models.Model(inputs=prev_frames, outputs=x)
    return seq_model