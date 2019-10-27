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

#############################################################################################################################################
###################################################### Recursive Autoencoder with skip connections & Exogenous Variables (time-flow, weather) + input in Decoder
#############################################################################################################################################
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


# try depth conv
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
#     return x 2x(convolution, batch normalization and Activation), max pooling and Dropout, with number of convolutions [16, 16x2, 16x4, 16x8, 16x8, 16x2], downsampling the image from 512 to 

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
    c5_1 = Dropout(dropout)(c5_1)
    
    c5_1_0 = conv2d_block(c5_1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, name='c6_2')
    c5_1 = MaxPooling2D(pool_size=(2, 2)) (c5_1_0)
    c5_1_a = Dropout(dropout)(c5_1)
    
    c5_1 = layers.Reshape(target_shape=(-1,), name='last')(c5_1_a)
    print("Encoder output shape:", c5_1_a.shape, "reshaped:", c5_1.shape)
    
    return Model(inputs=[input_img], outputs=[c5_1, c5_1_0, c5, c4, c3, c2, c1], name='encoder')#

def get_decoder(embedding, skip_connections, n_filters=16, dropout=0.5, batchnorm=True, channels_out=3):
    # load skip connections with shape: (16, 128), (32, 128), (64, 128), (128, 64), (256, 32), (512, 16))
    c5, c4, c3, c2, c1, c0, in_enc = skip_connections[0], skip_connections[1], skip_connections[2], skip_connections[3], \
                                     skip_connections[4], skip_connections[5], skip_connections[-1]
    
    # reshape to image  
    x = layers.Reshape((8, 8, 32))(embedding)
    print("Decoder input shape:", embedding.shape, "reshaped:", x.shape)
    
    # extra expansive path
    x = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same', name='u4') (x) # 16
    x = concatenate([x, c5])
    x = Dropout(dropout)(x)
    x = conv2d_block(x, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, name='c4')
    
    x = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same', name='u5') (x) # 32
    x = concatenate([x, c4])
    x = Dropout(dropout)(x)
    x = conv2d_block(x, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, name='c5')
    
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
    c9 = concatenate([c9, in_enc], name='concat_dec') # new concat with input
    outputs = conv2d_block(c9, n_filters=channels_out, kernel_size=3, batchnorm=batchnorm, name='out_dec')

    
    return Model(inputs=[embedding, c5, c4, c3, c2, c1, c0, in_enc], outputs=[outputs], name='decoder')


def get_encoder_decoder_w_input(embedding_size=2048, im_height=495, im_width=436, channels=3, 
                        dropout_enc=0.05, dropout_dec=0.05, b_norm_enc=True, b_norm_dec=True):

    # define inputs
    img_input_shape = (im_height, im_width, channels)
    input_img = Input(img_input_shape, name='img')
    input_emb = layers.Input((embedding_size,), name='embedding')
    
    skip_a = Input((16, 16, 32), name='skip_a')
    skip_0 = Input((32, 32, 128), name='skip_0')
    skip_1 = Input((64, 64, 128), name='skip_1')
    skip_2 = Input((128, 128, 64), name='skip_2')
    skip_3 = Input((256, 256, 32), name='skip_3')
    skip_4 = Input((512, 512, 16), name='skip_4')

    skip_connections_inputs = [skip_a, skip_0, skip_1, skip_2, skip_3, skip_4, input_img]
    
    # get encoder/decoder models
    with tf.name_scope('encoder'):
        encoder = get_encoder(input_img, n_filters=16, dropout=dropout_enc, batchnorm=b_norm_enc)
    with tf.name_scope('decoder'):
        decoder = get_decoder(input_emb, skip_connections_inputs, n_filters=16, dropout=dropout_dec, batchnorm=b_norm_dec)
    

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
        h, c5, c4, c3, c2, c1, c0 = encoder(current_frame)

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
        h, _, _, _, _, _, _ = encoder(current_fut_frame)

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
    embeddings = layers.TimeDistributed( layers.Dense(params['units_before_recurrent'], activation='relu', name='embedding_FC') )(embeddings)
    print("FC before recurrent embeddings.shape:", embeddings.shape)
    
    #################################################### Recurrent Phase
    # time encoder
    embeddings = layers.GRU(params['gru_enc_1'], activation='tanh', return_sequences = True, name='gru_enc_1_FC')(embeddings)
    embeddings = layers.GRU(params['gru_enc_2'], activation='tanh', return_sequences = False, name='gru_enc_2')(embeddings)

    embeddings = RepeatVector(length_seq_out, name='repeat_vector')(embeddings)

    # time decoder
    embeddings = layers.GRU(params['gru_dec_1'], activation='tanh', return_sequences = True, name='gru_dec_1')(embeddings)
    embeddings = layers.GRU(params['gru_dec_2'], activation='tanh', return_sequences = True, name='gru_dec_2')(embeddings)
    embeddings = layers.GRU(embedding_size, activation='tanh', return_sequences = True, name='gru_dec_3')(embeddings)
    print("recurrent embeddings.shape:", embeddings.shape)
    
    #################################################### Decoder Phase
    # get decoder for each frame predicted in the sequence (using skip connection from the most known recent frame and its inmput)
    prediced_frames = []
    for i in range(length_seq_out):
        # slice & decoder for current frame
        current_embedding = Lambda(lambda x: x[:, i], output_shape=(embedding_size,))(embeddings)
        current_pred_frame = decoder([current_embedding, c5, c4, c3, c2, c1, c0, current_frame])

        # append frames adding the sequence dimension to later concatenate them
        current_pred_frame = Reshape( (1,)+img_input_shape )(current_pred_frame)
        prediced_frames.append(current_pred_frame)
    prediced_frames = concatenate(prediced_frames, axis=1, name='Concat_predicted_frames')
    print("prediced_frames.shape:", prediced_frames.shape)
    
    return Model(inputs=[prev_frames, future_frames, day_input, time_input, weather_categorical_input, weather_continous_input], 
                 outputs=[prediced_frames], name='RAE_w_SC_WT_I'), embeddings, future_embeddings

def get_RAEwSC_and_WS_compiled_w_input(weights=None, length_seq_in=3, lr = 0.001, grad_clip=1., loss_weights={'predicted_frames':1., 'predicted_emb':1.}, 
                                 dropout_enc=0.05, dropout_dec=0.05, b_norm_enc=True, b_norm_dec=True, for_new_model=False):
    """ define and compile the model """
    
    params = {'units_before_recurrent': 2048, 'gru_enc_1': 256, 'gru_enc_2': 128, 'gru_dec_1': 128, 'gru_dec_2': 256, "embed_weather": 8}
    model, embeddings, future_embeddings = get_RAEwSC_and_ExogenousVars_w_input(params, length_seq_in=length_seq_in, 
                                           dropout_enc=dropout_enc, dropout_dec=dropout_dec, 
                                           b_norm_enc=b_norm_enc, b_norm_dec=b_norm_dec)
    
    if weights is not None:
        print('loading weitghs:', weights)
        model.load_weights(weights, by_name=True, skip_mismatch=True)
    
    if for_new_model:
        print("Freezing layers...")
        ###################################### change layer name 
        model.get_layer("Concat_predicted_frames").name = "Concat_predicted_frames_2"

        ###################################### freeze layers
        # once we load the weight we freeze all layers
        for i, layer in enumerate(model.layers):
            if 'norm' not in layer.name:
                layer.trainable = False

        # freeze also nested layers
        for i, layer in enumerate(model.get_layer("encoder").layers):
            if 'norm' not in layer.name:
                layer.trainable = False

        for i, layer in enumerate(model.get_layer("decoder").layers):
            if 'norm' not in layer.name:
                layer.trainable = False
        return model
    
    # compile with a custom loss that handle both regression and classification
    optimizer = [optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), optimizers.Adam(lr=lr, clipvalue=grad_clip)][-1]
    print("lr:", lr, 'optimizer:', optimizer)
    
    model.compile(optimizer=optimizer,
                  loss=loss_with_latent_term_2(future_embeddings, embeddings, loss_weights),
                  metrics=[losses.mean_squared_error, get_recurrent_embedding_loss(future_embeddings, embeddings, loss_weights)])
    
    return model

