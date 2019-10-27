import os
from keras import models, activations, losses, optimizers
from keras import backend as K 

import numpy as np

from src.data import get_generators, format_bytes, data_postprocess, data_preprocess, exchange_HEADING, vec2tensor
from src.data import write_data, create_directory_structure, data_2_submission_format

EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN, RAEwSCwWSwINwCLF = "RAE_w_SC_WS", "RAE_w_SC_WS_wIN", "RAEwSCwWSwINwCLF"

###################################################### LOSS DEFINITIONS    
def softmax_axis(axis=2):
    """ axis=2 refers to dim=5 in tensor [None, 3, 5, 495, 436]  """
    def soft(x):
        return activations.softmax(x, axis=axis)
    
    return soft

def cross_entropy_with_axis(axis_softmax=2):
	def ce_axis(y_true, y_pred):
		return K.categorical_crossentropy(y_true, y_pred, axis=axis_softmax)
	return ce_axis

def get_recurrent_embedding_loss(y_true_z, y_pred_z, loss_weights):
    
    def r_emb_loss(y_true, y_pred):
        """ Embedding loss """
        return loss_weights['predicted_emb']*losses.mean_squared_error(y_true_z, y_pred_z)
    
    return r_emb_loss

def loss_with_latent_term(y_true_z, y_pred_z, loss_weights={'predicted_frames':1., 'predicted_emb':1.}):
    """ returns a loss function that takes into account 2 terms:
        1. predicted vs. true loss
        2. latent space manifold
        
        input: 
            y_true_z : latent representation of future frames
            y_pred_z : latent prediction of current frames into the future
    """
    
    
    def seq2seq_recurrent_loss(y_true, y_pred):
        # Reconstruction loss
        predicted_frames_loss = losses.mean_squared_error(y_true, y_pred)
        
        # Embedding loss
        predicted_emb_loss = losses.mean_squared_error(y_true_z, y_pred_z)
        print('y_true:', y_true.shape, 'y_pred:', y_pred.shape)
        print('y_true_z:', y_true_z.shape, 'y_pred_z:', y_pred_z.shape)
        print('predicted_frames_loss:', predicted_frames_loss.shape, 'predicted_emb_loss:', predicted_emb_loss.shape)
        
        return loss_weights['predicted_frames']*predicted_frames_loss #+ loss_weights['predicted_emb']*predicted_emb_loss
    
    return seq2seq_recurrent_loss   

def loss_with_latent_term_2(y_true_z, y_pred_z, loss_weights):
    """ returns a loss function that takes into account 2 terms:
        1. predicted vs. true loss
        2. latent space manifold
        
        input: 
            y_true_z : latent representation of future frames
            y_pred_z : latent prediction of current frames into the future
    """
    
    
    def seq2seq_recurrent_loss(y_true, y_pred):
        # Reconstruction loss
        predicted_frames_loss = K.mean(losses.mean_squared_error(y_true, y_pred))
        
        # Embedding loss
        predicted_emb_loss = K.mean(losses.mean_squared_error(y_true_z, y_pred_z))
        
        return loss_weights['predicted_frames']*predicted_frames_loss + loss_weights['predicted_emb']*predicted_emb_loss
    
    return seq2seq_recurrent_loss

###################################################### METRIC DEFINITIONS
def MSE(x, y): 
    """ MSE pixel-wise, preserving time-slots and channels 
        input shape example: (48, 3, 3, 495, 436), first 3 is the number of time-slots, last three is the number of channels
        output shape example: (3, 3)
    """
    return np.mean((x-y)**2, axis=(0, -1, -2))

def add_info(mse):
    mean = np.vstack((np.asarray([['    Speed   ', '   Volume  ', '   Heading  ']]), mse.mean(axis=0)))
    mean = np.hstack((np.asarray([['        ', ' 5 minutes', '10 minutes', '15 minutes']]).T, mean))
    return mean

def print_eval(city, mse, mse_server, log_path):
    
    print('city:', city, 'shape:', mse.shape)
    print('mean mse:', mse.mean())
    print(add_info(mse))
    print('-----------------------------')
    print('mean mse like submission:', mse_server.mean())
    print(add_info(mse_server))
    
    # save all info as .npy
    np.save(log_path + city+'_mse_val.npy', mse)
    np.save(log_path + city+'_mse_server_val.npy', mse_server)
    
    # save mean by days info as csv for easy access
    np.savetxt(log_path + city+'_mse_val.csv', add_info(mse), delimiter=",", fmt='%s')
    np.savetxt(log_path + city+'_mse_server_val.csv', add_info(mse_server), delimiter=",", fmt='%s')

def save_y_hat(y, y_hat):
    np.save('/home/pherruzo/projects/nips_traffic/models/y', y)     
    np.save('/home/pherruzo/projects/nips_traffic/models/y_hat', y_hat)
    
def model_evaluate(dataset, model, log_path, city, model_type, mask_path='/home/pherruzo/projects/nips_traffic/models/'):
    
    # Use a mask to make zero areas where no road pass-through
    binary_mask = np.moveaxis(np.load(mask_path+city+'_mask.npy'), -1, -3) # channels first
    binary_mask = np.expand_dims(np.expand_dims(binary_mask, axis=0), axis=0) # create dimension for samples and time-slots
    
    city_days_mse, city_days_mse_server = [], []
    city_days_mse_server_clf, heading_acc, heading_acc_clf = [], [], []
    
    total_batches = len(dataset)
    conv_and_clf, exchange_heading_for_last_seen_heading = "ConvLSTM+Clf", False
    clf_as_heading = False
    
    for i, sample in enumerate(dataset):
        print("Evaluating time-bin/time-slot batch {}/{}".format(i+1, total_batches))
        
        ######## load data
        if model_type in ["ConvLSTM", "ConvLSTM+Clf"]:
            x, y, sample_weights = sample[0], sample[1]['convlstm_3'], sample[2]
        elif model_type in [EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN ]:#=='RAE_w_SC_WS':
            x, y, sample_weights = sample[0], sample[1]['Concat_predicted_frames'], sample[2]
            x['future_frames'] = x['prev_frames'][:, -3:] # take the last three
        elif model_type in [RAEwSCwWSwINwCLF]:
            x, y, sample_weights = sample[0], sample[1], sample[2]
            x['future_frames'] = x['prev_frames'][:, -3:] # take the last three
            
            y_clf = y['softmax_clf']
            y = y['Concat_predicted_frames']
            clf_as_heading = True
        else:
            x, y, sample_weights = sample[0]['prev_frames'], sample[1]['Concat_predicted_frames'], sample[2]
        
        # extra data depending on model
        if model_type == conv_and_clf: #ConvLSTM
            y_clf = sample[1]['softmax_clf']
            clf_as_heading = True

        ######## predict and compute mse
        if model_type == conv_and_clf: #ConvLSTM
            y_hat, y_hat_clf = model.predict(x)
            
        elif model_type == 'RAE_w_SC':
            y_hat = model.predict([x, x])
            
            # put in the same shape as all models
            y_hat = np.transpose(y_hat, (0, 1, 4, 2, 3))
            y = np.transpose(y, (0, 1, 4, 2, 3))
            
        elif model_type in [EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN ]:#=='RAE_w_SC_WS':
            y_hat = model.predict(x)
            
            # put in the same shape as all models (sadly it means channel first)
            y_hat = np.transpose(y_hat, (0, 1, 4, 2, 3))
            y = np.transpose(y, (0, 1, 4, 2, 3))
            
        elif model_type in [RAEwSCwWSwINwCLF]:
            y_hat, y_hat_clf = model.predict(x)
            
            # put in the same shape as all models (sadly it means channel first)
            y_hat = np.transpose(y_hat, (0, 1, 4, 2, 3))
            y = np.transpose(y, (0, 1, 4, 2, 3))
        else:
            y_hat = model.predict(x)
        
        if exchange_heading_for_last_seen_heading:
            # from the last frame in the sequence, get heading
            heading_last_frame = x[:, -1, -1]
            heading_last_frame = np.moveaxis(np.array([heading_last_frame, heading_last_frame, heading_last_frame]), 0, 1)

            # assign it as the heading for all predicted frames
            y_hat[:, :, -1] = heading_last_frame

        mse = MSE(y_hat, y)
        city_days_mse.append(mse)
        
        ###################### imitate evaluation in server to compute mse
        # 1. save output in range 0, 255 (integer)
        y_hat = data_postprocess(y_hat, binary_mask)
        #save_y_hat(y, y_hat)
        # 1. load input and rescale to 0, 1 (float)
        y_hat = data_preprocess(y_hat)
        
        mse = MSE(y_hat, y)
        city_days_mse_server.append(mse)
        
        # compute acc for the heading channel
        heading_acc.append((np.sum(y[:,:,-1]==y_hat[:,:,-1]))/y[:,:,-1].size)
#         print("Acc y_hat:", np.sum(y[:,:,-1]==y_hat[:,:,-1])/y[:,:,-1].size, "rmse:", np.mean((y[:,:,-1]-y_hat[:,:,-1])**2), mse.mean())
        
        ###################### compute mse with clf
        if clf_as_heading: #ConvLSTM
            # transform HEADING vector to image and exchange HEADING dimension
            y_hat_clf = vec2tensor(y_hat_clf)
            y_hat_clf = exchange_HEADING(y_hat.copy(), y_hat_clf)
            
            mse = MSE(y_hat_clf, y)
            city_days_mse_server_clf.append(mse)
            
            # compute acc for the heading channel
            heading_acc_clf.append((np.sum(y[:,:,-1]==y_hat_clf[:,:,-1]))/y_hat_clf[:,:,-1].size)
#             print("Acc y_hat_clf:", np.sum(y[:,:,-1]==y_hat_clf[:,:,-1])/y_hat_clf[:,:,-1].size, "rmse:", np.mean((y[:,:,-1]-y_hat_clf[:,:,-1])**2), mse.mean())
#         return y_hat, y_hat_clf, x, y, y_clf
    
    # convert arrays to numpy
    city_days_mse, city_days_mse_server = np.asarray(city_days_mse), np.asarray(city_days_mse_server)    
    print_eval(city, city_days_mse, city_days_mse_server, log_path)
    
    if clf_as_heading:
        city_days_mse_server_clf = np.asarray(city_days_mse_server_clf)
        print('-----------------------------')
        print('mean mse with HEADING as clf:', city_days_mse_server_clf.mean())
        print(add_info(city_days_mse_server_clf))
    
    print("=========") # we compute average of average since all batches have the same number of samples (mb except the last one)
    print("Acc y_hat in HEADING:", np.asarray(heading_acc).mean())
    if clf_as_heading:
        print("Acc y_hat_clf in HEADING:", np.asarray(heading_acc_clf).mean())

###################################################### OUTPUT FILES GENERATION
def write_submission_files(dataset, model, output_path, city, model_type, mask_path='/home/pherruzo/projects/nips_traffic/models/'):
    
    create_directory_structure(output_path, city)
    
    # Use a mask to make zero areas where no road pass-through
    binary_mask = np.moveaxis(np.load(mask_path+city+'_mask.npy'), -1, -3) # channels first
    binary_mask = np.expand_dims(np.expand_dims(binary_mask, axis=0), axis=0) # create dimension for samples and time-slots
    
    # params
    total_batches = len(dataset)
    conv_and_clf, exchange_heading_for_last_seen_heading, use_clf_as_heading = "ConvLSTM+Clf", False, False
    
    for i, sample in enumerate(dataset):
        # get name of the file and data
        f = sample[1]
        sample = sample[0]
            
        # 1. load data
        if model_type in ["ConvLSTM", "ConvLSTM+Clf"]:
            x, y, sample_weights = sample[0], sample[1]['convlstm_3'], sample[2]
        elif model_type in [EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN ]:#=='RAE_w_SC_WS':
            x, y, sample_weights = sample[0], sample[1]['Concat_predicted_frames'], sample[2]
            x['future_frames'] = x['prev_frames'][:, -3:] # take the last three
        elif model_type in [RAEwSCwWSwINwCLF]:
            x, y, sample_weights = sample[0], sample[1], sample[2]
            x['future_frames'] = x['prev_frames'][:, -3:] # take the last three
            
            y_clf = y['softmax_clf']
            y = y['Concat_predicted_frames']
            clf_as_heading = True
        else:
            x, y, sample_weights = sample[0]['prev_frames'], sample[1]['Concat_predicted_frames'], sample[2]
        # extra data depending on model
        if model_type == conv_and_clf: #ConvLSTM
            y_clf = sample[1]['softmax_clf']
            clf_as_heading = True

        # 2. predict and compute mse
        if model_type == conv_and_clf: #ConvLSTM
            y_hat, y_hat_clf = model.predict(x)
        elif model_type == 'RAE_w_SC':
            y_hat = model.predict([x, x])
            
            # put in the same shape as all models
            y_hat = np.transpose(y_hat, (0, 1, 4, 2, 3))
            y = np.transpose(y, (0, 1, 4, 2, 3))
        elif model_type in [EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN ]:#=='RAE_w_SC_WS':
            y_hat = model.predict(x)
            
            # put in the same shape as all models (sadly it means channel first)
            y_hat = np.transpose(y_hat, (0, 1, 4, 2, 3))
            y = np.transpose(y, (0, 1, 4, 2, 3))
        elif model_type in [RAEwSCwWSwINwCLF]:
            y_hat, y_hat_clf = model.predict(x)
            
            # put in the same shape as all models (sadly it means channel first)
            y_hat = np.transpose(y_hat, (0, 1, 4, 2, 3))
            y = np.transpose(y, (0, 1, 4, 2, 3))
        else:
            y_hat = model.predict(x)
        
        ###################### different heading predictions
        if exchange_heading_for_last_seen_heading:
            # from the last frame in the sequence, get heading
            heading_last_frame = x[:, -1, -1]
            heading_last_frame = np.moveaxis(np.array([heading_last_frame, heading_last_frame, heading_last_frame]), 0, 1)

            # assign it as the heading for all predicted frames
            y_hat[:, :, -1] = heading_last_frame
            print("Using last known heading for prediction")
        if use_clf_as_heading: #ConvLSTM
            # transform HEADING vector to image and exchange HEADING dimension
            y_hat_clf = vec2tensor(y_hat_clf)
            y_hat = exchange_HEADING(y_hat.copy(), y_hat_clf)
            print("Using clf as heading")
            
        # 3. transform data into submission format
        y_hat = data_2_submission_format(y_hat, binary_mask)

        # 4. generate output file path
        outfile = os.path.join(output_path, city, city+'_test', f.split('/')[-1])
        write_data(y_hat, outfile)
        print("City:{}, just wrote file {}/{}: {}".format(city, i+1, total_batches, outfile))
        
def write_submission_files_bu(dataset, model, output_path, city, model_type, mask_path='/home/pherruzo/projects/nips_traffic/models/'):
    
    create_directory_structure(output_path, city)
    
    # Use a mask to make zero areas where no road pass-through
    binary_mask = np.moveaxis(np.load(mask_path+city+'_mask.npy'), -1, -3) # channels first
    binary_mask = np.expand_dims(np.expand_dims(binary_mask, axis=0), axis=0) # create dimension for samples and time-slots
    
    # params
    total_batches = len(dataset)
    conv_and_clf, exchange_heading_for_last_seen_heading, use_clf_as_heading = "ConvLSTM+Clf", False, False
    
    for i, sample in enumerate(dataset):
        # get name of the file and data
        f = sample[1]
        sample = sample[0]
            
        # 1. load data
        if model_type in ["ConvLSTM", "ConvLSTM+Clf"]:
            x, y, sample_weights = sample[0], sample[1]['convlstm_3'], sample[2]
        elif model_type in [EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN ]:#=='RAE_w_SC_WS':
            x, y, sample_weights = sample[0], sample[1]['Concat_predicted_frames'], sample[2]
            x['future_frames'] = x['prev_frames'][:, -3:] # take the last three
        else:
            x, y, sample_weights = sample[0]['prev_frames'], sample[1]['Concat_predicted_frames'], sample[2]
        # extra data depending on model
        if model_type == conv_and_clf: #ConvLSTM
            y_clf = sample[1]['softmax_clf']

        # 2. predict and compute mse
        if model_type == conv_and_clf: #ConvLSTM
            y_hat, y_hat_clf = model.predict(x)
        elif model_type == 'RAE_w_SC':
            y_hat = model.predict([x, x])
            
            # put in the same shape as all models
            y_hat = np.transpose(y_hat, (0, 1, 4, 2, 3))
            y = np.transpose(y, (0, 1, 4, 2, 3)) 
        elif model_type in [EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN ]:#=='RAE_w_SC_WS':
            y_hat = model.predict(x)
            
            # put in the same shape as all models (sadly it means channel first)
            y_hat = np.transpose(y_hat, (0, 1, 4, 2, 3))
            y = np.transpose(y, (0, 1, 4, 2, 3))
        else:
            y_hat = model.predict(x)
        
        ###################### different heading predictions
        if exchange_heading_for_last_seen_heading:
            # from the last frame in the sequence, get heading
            heading_last_frame = x[:, -1, -1]
            heading_last_frame = np.moveaxis(np.array([heading_last_frame, heading_last_frame, heading_last_frame]), 0, 1)

            # assign it as the heading for all predicted frames
            y_hat[:, :, -1] = heading_last_frame
            print("Using last known heading for prediction")
        if use_clf_as_heading: #ConvLSTM
            # transform HEADING vector to image and exchange HEADING dimension
            y_hat_clf = vec2tensor(y_hat_clf)
            y_hat = exchange_HEADING(y_hat.copy(), y_hat_clf)
            print("Using clf as heading")
            
        # 3. transform data into submission format
        y_hat = data_2_submission_format(y_hat, binary_mask)

        # 4. generate output file path
        outfile = os.path.join(output_path, city, city+'_test', f.split('/')[-1])
        write_data(y_hat, outfile)
        print("City:{}, just wrote file {}/{}: {}".format(city, i+1, total_batches, outfile))
        
def write_submission_files_backup(dataset, model, output_path, city, model_type, mask_path='/home/pherruzo/projects/nips_traffic/models/'):
    
    create_directory_structure(output_path, city)
    
    # Use a mask to make zero areas where no road pass-through
    binary_mask = np.moveaxis(np.load(mask_path+city+'_mask.npy'), -1, -3) # channels first
    binary_mask = np.expand_dims(np.expand_dims(binary_mask, axis=0), axis=0) # create dimension for samples and time-slots
    
    # params
    total_batches = len(dataset)
    conv_and_clf, exchange_heading_for_last_seen_heading, use_clf_as_heading = "ConvLSTM+Clf", False, False
    
    for i, sample in enumerate(dataset):
        # get name of the file and data
        f = sample[1]
        sample = sample[0]
        
        # 1. load data
        x, y, sample_weights = sample[0], sample[1]['convlstm_3'], sample[2]
        if model_type == conv_and_clf:
            y_clf = sample[1]['softmax_clf']

        # 2. predict
        if model_type == conv_and_clf: #ConvLSTM
            y_hat, y_hat_clf = model.predict(x)
        else:
            y_hat = model.predict(x)
        
        ###################### different heading predictions
        if exchange_heading_for_last_seen_heading:
            # from the last frame in the sequence, get heading
            heading_last_frame = x[:, -1, -1]
            heading_last_frame = np.moveaxis(np.array([heading_last_frame, heading_last_frame, heading_last_frame]), 0, 1)

            # assign it as the heading for all predicted frames
            y_hat[:, :, -1] = heading_last_frame
            print("Using last known heading for prediction")
        if use_clf_as_heading: #ConvLSTM
            # transform HEADING vector to image and exchange HEADING dimension
            y_hat_clf = vec2tensor(y_hat_clf)
            y_hat = exchange_HEADING(y_hat.copy(), y_hat_clf)
            print("Using clf as heading")
            
        # 3. transform data into submission format
        y_hat = data_2_submission_format(y_hat, binary_mask)

        # 4. generate output file path
        outfile = os.path.join(output_path, city, city+'_test', f.split('/')[-1])
        write_data(y_hat, outfile)
        print("City:{}, just wrote file {}/{}: {}".format(city, i+1, total_batches, outfile))