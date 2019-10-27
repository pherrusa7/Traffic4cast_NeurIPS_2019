import os
import time
import sys

import random
import h5py
import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from keras.utils import Sequence
from sklearn.utils import class_weight
from keras.utils.np_utils import to_categorical 

import src.exogenous_data as exogenous_data

###################################################### Data Paths
DATA = '/home/pherruzo/data/nips_traffic/'
cities = {'Berlin': 'Berlin/Berlin', 'Istanbul': 'Istanbul/Istanbul', 'Moscow': 'Moscow/Moscow'} #[4.44, 7.17, 11.02] GB respectively
sets = {'training': '_training/' , 'validation': '_validation/', 'test': '_test/'}
NUM_SLOTS_5_MIN = 288
NUM_IN_SEQ_TEST = 12

# The following indicies are the start indicies of the 3 images to predict in the 288 time bins (0 to 287)
# in each daily test file. These are time zone dependent. Berlin lies in UTC+2 whereas Istanbul and Moscow
# lie in UTC+3.
utcPlus2 = [30, 69, 126, 186, 234]
utcPlus3 = [57, 114, 174,222, 258]
utcPlus2 = [n-NUM_IN_SEQ_TEST for n in utcPlus2] # set the idxs at the beggining of the blocks of 12 bins informed
utcPlus3 = [n-NUM_IN_SEQ_TEST for n in utcPlus3]
TEST_SLOTS = {'Berlin': utcPlus2, 'Istanbul': utcPlus3, 'Moscow': utcPlus3}

# Some variables
EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN, RAEwSCwWSwINwCLF = "RAE_w_SC_WS", "RAE_w_SC_WS_wIN", "RAEwSCwWSwINwCLF"


ex_train = '20180418_100m_bins.h5'
ex_test = '20180717_100m_bins.h5'
n_file = '_100m_bins.h5'

# DEBUG = True

def format_bytes(size):
    # function from https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb/37423778
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'

###################################################### FILES RETRIEVAL

def get_files_in_folder(path):
    files = []
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.h5' in file:
                files.append(os.path.join(r, file))
    return files

def get_city_files_dict(city=None):
    """ retrieve all files per one or all cities
    
        returns a dict with keys 'training', 'validation' and 'test'
    """
        
    paths = {}
    
    if city is None:
        city_keys = cities.keys()
    else:
        city_keys = [city]
        
    for sub_set in sets.keys():
        paths[sub_set] = []
        for city in city_keys:
            path = DATA + cities[city] + sets[sub_set]
            files = get_files_in_folder(path)
            
            paths[sub_set] += files
        
    return paths

###################################################### FILES CREATION
def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()

def create_directory_structure(root, city):
    city_dirs = {'Berlin': os.path.join(root, "Berlin","Berlin_test"),
                 'Istanbul': os.path.join(root, "Istanbul","Istanbul_test"),
                 'Moscow': os.path.join(root, "Moscow", "Moscow_test")}[city]

    try:
        os.makedirs(city_dirs)
    except OSError:
        print("failed to create directory structure in", city_dirs)
        sys.exit(2)

###################################################### DATA PROCESSING
def data_postprocess(data, binary_mask):
    """ rescale data to [0, 255], rounds it and cast it to type uint8 """

    # clip outputs between 0, 1
    data = np.clip(data, 0, 1)
    
    # rescale to [0, 255]
    data *= 255
    
    # make all number integers
    data = np.around(data)
    data = data.astype(np.uint8)
    
    # Use a mask to make zero areas where no road pass-through
    data = data*binary_mask
    
    if False: # makes performance worse
        # map Heading to 1 of the 4 options (+ option 0, which is already set)
        conditions = [[0, 42, 1], [42, 126, 85], [126, 210, 170], [210, 255, 255]]

        for condA, condB, val in conditions:
            #print(condA, "< X <= ", condB, "---->", val) # debug
            data[:,:,-1,:,:] = np.where((condA < data[:,:,-1,:,:]) & (data[:,:,-1,:,:] <= condB), val , data[:,:,-1,:,:])

    return data

def data_2_submission_format(data, binary_mask, model_type="other"):
    """ rescale data to [0, 255], rounds it and cast it to type uint8 """
    
    # postprocess predicted data
    data = data_postprocess(data, binary_mask)
    
    # transpose to (samples, timesteps, rows, columns, channels)
    # from         (samples, timesteps, channels, rows, columns)
    if model_type != 'RAE_w_SC':
        data = np.moveaxis(data, -3, -1)

    return data

def data_preprocess(data):
    """ cast data to float and rescale """
    
    data = data.astype(np.float32)
    data /= 255.
    return data

def next_batch(data, num_out_seq, sparse_loss, model_type, sub_set, extra_vars={}):
    """ Last step of preprocessing depends on the model that we are using 
        model_type in ["ConvLSTM", "ConvLSTM+Clf", "RAE_w_SC", EXTRA_DATA_MODEL]
        
        It depends on the model that we are using since they can have diff num of inputs/outputs
    """
    
    # transpose to (samples, timesteps, channels, rows, columns)
    if model_type in ["ConvLSTM", "ConvLSTM+Clf"] :
        data = np.transpose(data, (0, 1, 4, 2, 3))
        
    if model_type=="ConvLSTM+Clf":
        # make heading variable discrete for classification problem (the "-1" bellow is for the HEADING channel)
        # i.e., transform 'heading' from {0, 1, 85, 170, 255} to {0,1,2,3,4}
        y_clf = categorical_or_sparse_loss(data[:, -num_out_seq:, -1], sparse_loss)

    elif model_type==RAEwSCwWSwINwCLF:
        # make heading variable discrete for classification problem (the "-1" bellow is for the HEADING channel)
        # i.e., transform 'heading' from {0, 1, 85, 170, 255} to {0,1,2,3,4}
        y_clf = categorical_or_sparse_loss(data[:, -num_out_seq:, :, :, -1], sparse_loss, model_type)

    # preprocess data
    data = data_preprocess(data)
    
    # assign
    x = data[:, :-num_out_seq]
    if model_type=="ConvLSTM+Clf":
        y = {'convlstm_3': data[:, -num_out_seq: ], 'softmax_clf': y_clf}
        
        if sub_set=="test": # in test time target has only the value 0
            unique_values=[0]
        else:
            unique_values=[0, 1, 2, 3, 4]
        sample_weights = {'softmax_clf': get_sample_weights(y_clf, unique_values=unique_values, d_class_weights=None)}
        return x, y, sample_weights
    
    elif model_type=="ConvLSTM":
        y = {'convlstm_3': data[:, -num_out_seq: ]}
        return x, y, None
    
    elif model_type=="RAE_w_SC":
        y = {'Concat_predicted_frames': data[:, -num_out_seq: ]}
        
        if sub_set=="test": # in test we don't have future frames
            x = {'prev_frames': x, 'future_frames': x}
        else:
            x = {'prev_frames': x, 'future_frames': data[:, -num_out_seq:]}
        return x, y, None
    
    elif model_type in [EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN]:
        y = {'Concat_predicted_frames': data[:, -num_out_seq: ]}
        
        if sub_set=="test": # in test we don't have future frames    'weather_categorical', 'weather_continous'
            x = {'prev_frames': x, 'future_frames': x,
                'day_info': extra_vars['day_info'], 'time_info': extra_vars['time_info'],
                'weather_categorical': extra_vars['weather_categorical'], 'weather_continous': extra_vars['weather_continous']}
        else:
            x = {'prev_frames': x, 'future_frames': data[:, -num_out_seq:],
                'day_info': extra_vars['day_info'], 'time_info': extra_vars['time_info'],
                'weather_categorical': extra_vars['weather_categorical'], 'weather_continous': extra_vars['weather_continous']}
        return x, y, None
    
    elif model_type == RAEwSCwWSwINwCLF:
        y = {'Concat_predicted_frames': data[:, -num_out_seq: ],
             'softmax_clf': y_clf}
        
        if sub_set=="test": # in test we don't have future frames    'weather_categorical', 'weather_continous'
            x = {'prev_frames': x, 'future_frames': x,
                'day_info': extra_vars['day_info'], 'time_info': extra_vars['time_info'],
                'weather_categorical': extra_vars['weather_categorical'], 'weather_continous': extra_vars['weather_continous']}
        else:
            x = {'prev_frames': x, 'future_frames': data[:, -num_out_seq:],
                'day_info': extra_vars['day_info'], 'time_info': extra_vars['time_info'],
                'weather_categorical': extra_vars['weather_categorical'], 'weather_continous': extra_vars['weather_continous']}
        
        # prepare sample weights. This time sample weight will force de model to punish more predictions far way from the real value
        if sub_set=="test": # in test time target has only the value 0
            unique_values=[0]
        else:
            unique_values=[0, 1, 2, 3, 4]
        sample_weights = {'softmax_clf': get_sample_weights(y_clf, unique_values=unique_values, d_class_weights=None)}
        
        return x, y, sample_weights
    else:
        sys.exit('Error: {} not defined in next_batch(...)'.format(model_type))

###################################################### DATA RESHAPE - Changes depending on sparse_cross_E/categorical_cross_E

def check_equals(y_o, y_p):
    if np.sum(y_p==y_o) / y_o.size == 1:
        print("--> Yeah, they are the same :)")
    else:
        print("WARNING: It's not the identity")
        
def vec2tensor(y_hat):
    # reshape prediction to original size
    y_hat_ori = np.reshape(y_hat, (y_hat.shape[0], 3, 495, 436, 5))

    # channels first
    y_hat_ori_sort = np.transpose(y_hat_ori, (0, 1, 4, 2, 3))
    
#     print(1, y_hat.shape, 2, y_hat_ori.shape, 3, y_hat_ori_sort.shape)
    return y_hat_ori_sort

def tensor2vec(y_1hot):
    # channels last
    y_1hot_sort = np.transpose(y_1hot, (0, 1, 3, 4, 2))

    # reshape
    y_1hot_sort_vect = np.reshape(y_1hot_sort, (y_1hot_sort.shape[0], -1, 5)) # vectorize all frames

    print(1, y_1hot.shape, 2, y_1hot_sort.shape, 3, y_1hot_sort_vect.shape)
    return y_1hot_sort_vect

###################################################### DATA IMBALANCE SOLUTIONS

def find_sample_weights(Y, unique_values):
    class_weights = class_weight.compute_class_weight('balanced', unique_values, Y.ravel())
    d_class_weights = dict(enumerate(class_weights))
    
    return d_class_weights


def get_sample_weights(y_true, unique_values=[0, 1, 2, 3, 4],
                       d_class_weights={0: 0.21411693, 1: 12.23366874, 2: 11.65751912, 3: 13.36453063, 4: 11.45387603}):
    """ Get weight for your imbalanced sample. 
            · If you set d_class_weights=None, weights will be computed for current sample <-- more accurate, slow ~5s
            · If not, precomputed weights for the whole dataset will be use <-- faster ~254 ms ~1/4s <-- 20x faster
        Suggestion, if using fit_generator, which computes batches in parallel to computations, use d_class_weights=None,
        otherwise, stick to general weights if you don't want this to be a bottleneck
    """
    y_true = np.squeeze(y_true, axis=-1)
    
    if d_class_weights is None:
        d_class_weights = find_sample_weights(y_true, unique_values)
#     print(d_class_weights)
    sample_weights = np.ones_like(y_true).astype(np.float32)
    for v in unique_values:
        sample_weights[y_true==v] = d_class_weights[v]
        
    return sample_weights

###################################################### Prepare data channel HEADING for classification
def getval_array(d):
    """ fast method to swap numbers in dict d (found in stackoverflow: todo: find the link) """
    v = np.array(list(d.values()))
    k = np.array(list(d.keys()))
    maxv = k.max()
    minv = k.min()
    n = maxv - minv + 1
    val = np.empty(n,dtype=v.dtype)
    val[k] = v
    return val

def reg2clf(y, swap={0:0, 1:1, 85:2, 170:3, 255:4}, model_type=None):
    """ transforms any finite set of numbers to another """
    
    # swap values
    val_arr = getval_array(swap)
    values = val_arr[y]
    
    # make dummy dimensions and locate them appropriately
    categorical_labels = to_categorical(values, num_classes=5)
    if model_type is None:
        categorical_labels = np.transpose(categorical_labels, (0, 1, 4, 2, 3))
    
    #print(np.unique(y), np.unique(val_arr[y]))
    return categorical_labels

def reg2clf_vectorized(y, swap={0:0, 1:1, 85:2, 170:3, 255:4}):
    """ transforms any finite set of numbers to another """
    
    val_arr = getval_array(swap)
    values = val_arr[y]
    
    # reshape values to vector
    values = values.reshape((values.shape[0], -1))
    
#     print(np.unique(y), np.unique(y/255.), np.unique(val_arr[y]), values.shape, y.shape)
    return values

def exchange_HEADING(y_pred, y_pred_clf, swap={0:0, 1:1/255., 2:85/255., 3:170/255., 4:255/255.}):
    """  This methods take the dimension of HEADING in y_pred away and makes a replacement with the 
         corresponding one in y_pred_clf.
    """
    # 1. get predictions from the HEADING classification branch
    y_pred_clf = np.argmax(y_pred_clf, axis=2)
    #print(y_pred_clf.shape, np.unique(y_pred_clf))

    # 2. translate argument to number representing HEADING-direction using swap
    val_arr = getval_array(swap)
    y_pred_clf = val_arr[y_pred_clf]
    #print(y_pred_clf.shape, np.unique(y_pred_clf))

    # 3. exchange values of HEADING channel
    #print('before:', y_pred.shape, np.unique(y_pred[:,:,-1,:,:]))
    y_pred[:,:,-1,:,:] = y_pred_clf
    #print('after:', y_pred.shape, np.unique(y_pred[:,:,-1,:,:]))
    return y_pred

def categorical_or_sparse_loss(y, sparse, model_type=None):
    y = reg2clf_vectorized(y.astype(int))
    y = np.expand_dims(y, axis=-1)

    if not sparse:
        y = to_categorical(y.astype(int))
    return y

###################################################### DATA GENERATOR
class Traffic_Dataset(Sequence):
    """ 
    ARGS:    
        · mode: Set the number of samples available to the system for training. If mode="all_possible_slots",
        we will use all possible sequences to train in a day, sliding each bin 1 time when creating the 
        sequences. If mode="non-overlapping", batches will be loaded by consecutive non-overlapping 
        sequences, e.g., if num_in_seq=3, then lenght_seq = 3+3=6, so each day will have 48 non-overlapping 
        sequences to train of lenght 6 (6*48=285 time bins of 5 min in a day). 
        data_split = ["non-overlapping", "all_possible_slots", "test"]
        
        · sub_set: trainin/validation/test
        
        · model_type: selects how to preprocess data depending on the model we will use
        
        · loss_type: if model_type="ConvLSTM+Clf", then loss can either be 'Sparse' or 'Categorical' for the
        Heading channel
        
        · city: If 'city' is not provided, 'test' subset can not be retrieve, since the methods requires to know 
        which time-slots are needed, and this depends on the city
        
        Some blogs:
        · https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
    """
    def __init__(self, batch_size, model_type="ConvLSTM+Clf", num_in_seq=12, num_out_seq=3, loss_type='Sparse',
                 sub_set='training', city=None, num_time_slots=NUM_SLOTS_5_MIN, data_split="all_possible_slots",
                debug=False):
        
        self.sub_set = sub_set
        self.mode = data_split
        self.city = city
        self.model_type = model_type
        
        # each sample in batch X is a sequence of samples (of lenght 'self.num_in_seq')
        self.num_in_seq = num_in_seq
        self.num_out_seq = num_out_seq
        self.lenght_seq = self.num_in_seq + self.num_out_seq
        self.batch_size = batch_size
        
        # each day has 'num_time_slots' of frames
        self.slots_per_day = num_time_slots
        
        # load paths for all cities: e.g. paths[<'training', 'validation', 'test'>]
        self.file_paths = get_city_files_dict(city)[sub_set]
        if debug:
            self.file_paths = self.file_paths[:1]
            
        # create all slots for each day to easy access them while training
        # the size of the dataset is: num_days * (time_slots_per_day-self.lenght_seq+1)
        # Note (time_slots_per_day-self.lenght_seq+1) is 288-15+1. This way we ansure that the last sequence
        # to get from a day starts at the slot 273, so the sequence is of length 15
        self.paths_wt_time_slots = [[path, i] for path in self.file_paths for i in self.__get_slots(self.mode)]
        
        # set if loss is sparse or not, so y_hat can be converted
        if loss_type is None:
            loss_type = 'Sparse'
        self.sparse_loss = True
        if loss_type != 'Sparse':
            self.sparse_loss = False
        
        # shuffle if it's training, if not, this method does nothing
        self.on_epoch_end()
        if debug:
            self.paths_wt_time_slots = self.paths_wt_time_slots[:16]
        
        # prepare an offset for batch retrieval when its test
        self.offset_if_test = 0
        if self.mode == "like-test" or self.sub_set=="test":
            self.offset_if_test = NUM_IN_SEQ_TEST - self.num_in_seq
            
        if self.sub_set=="test":
            self.batch_size = len(TEST_SLOTS[self.city]) # there is 5 time-slots to predict each day (we force day=batch) to generate final preds to submit
        
        if self.model_type in [EXTRA_DATA_MODEL, EXTRA_DATA_MODELwIN, RAEwSCwWSwINwCLF]:
            self.time_flow_dict = exogenous_data.get_time_vector(self.city, self.sub_set)
            self.weather = exogenous_data.get_weather_df(self.city)
        else:
            self.time_flow_dict = None
            self.weather = None
        
    def __len__(self):
        """ number of iterations to complete 1 epoch """
        if self.sub_set != 'training':
            return int(np.ceil(len(self.paths_wt_time_slots) / float(self.batch_size)))
#             return 3 # if debugging
        else:
            return int(np.ceil(len(self.paths_wt_time_slots) / float(self.batch_size)))
#             return int(int(np.ceil(len(self.paths_wt_time_slots) / float(self.batch_size)))*0.17) # to make each epoch shorter
    
    def __getitem__(self, idx):
#         print("Already preparing batch idx {} in set {}".format(idx, self.sub_set))
        # get next batch IDs
        batch = self.paths_wt_time_slots[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # Load each single sequence in the batch
        data = np.array([self.__get_frames_seq(path, i + self.offset_if_test, self.lenght_seq) for path, i in batch])
        
        # add exogenous variables 
        extra_vars = {}
        if self.time_flow_dict is not None: # self.offset_if_test should be alway 9 to reuse models even if (num_in_seq != 3)
            extra_vars =  exogenous_data.get_batch_exogenous_info(batch, self.lenght_seq, offset_if_test=self.offset_if_test, time_dict=self.time_flow_dict, df_weather=self.weather)
            # keys: 'day_info', 'time_info', 'weather_categorical', 'weather_continous'
        
        if self.sub_set == "test":
            paths = np.array([path for path, i in batch])
            assert len(np.unique(paths))==1, "ERROR: Test batch contains + than 1 day"
            return (next_batch(data, self.num_out_seq, self.sparse_loss, self.model_type, self.sub_set, extra_vars), paths[0])
        
        return next_batch(data, self.num_out_seq, self.sparse_loss, self.model_type, self.sub_set, extra_vars)
    
    def __get_frames_seq(self, path, idx, length):
        f = h5py.File(path, 'r')
        key = list(f.keys())[0]

        data = f[key][idx:idx+length]
        return data
    
    def __get_slots(self, mode):
        """ define the slots in a day to train or test depending on the selected mode """
        
        if mode == "all_possible_slots": # 0, 1, 2, ..., N-(lenght-1)
            slots = list(range(self.slots_per_day-self.lenght_seq+1))
        if mode == "non-overlapping": # 0-5, 6-11, 12-17, 18-23, ...
            slots = list(range(0, NUM_SLOTS_5_MIN-self.lenght_seq+1, self.lenght_seq))
        if mode == "like-test": # 0-5, 6-11, 12-17, 18-23, ...
            slots = TEST_SLOTS[self.city]
        
        # if it is the test subset, we only need to predict at certain time-steps
        if self.sub_set == "test": # Only test index
            slots = TEST_SLOTS[self.city]

        return slots
    
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        if self.sub_set == "training":
            print("-->> Data has been shuffled in:", self.sub_set)
            random.shuffle(self.paths_wt_time_slots)

            
def get_generators(model_type, bs_tra, city, lenght_seq_in, bs_val, data_split, bs_test=5, debug=False, loss_type=None):
    """ get training/validation/test generators """
    
    val_split = "non-overlapping" # test faster than 'all_possible_slots'
    if data_split == "like-test":
        val_split = data_split
        
    tra_ds = Traffic_Dataset(bs_tra, city=city, num_in_seq=lenght_seq_in, model_type=model_type, data_split=data_split, loss_type=loss_type, debug=debug)
    val_ds = Traffic_Dataset(bs_val, city=city, num_in_seq=lenght_seq_in, model_type=model_type, data_split=val_split, sub_set='validation', loss_type=loss_type, debug=debug)
    tes_ds = Traffic_Dataset(bs_test, city=city, num_in_seq=lenght_seq_in, model_type=model_type, data_split=data_split, sub_set='test', loss_type=loss_type)
    
    return tra_ds, val_ds, tes_ds

""" Examples of iterating a generator
1)
x,y, yclf = training_ds.__getitem__(0)
print(x.shape,y.shape, x.sum())
print(format_bytes(x.nbytes), format_bytes(y.nbytes), format_bytes(1.*x.nbytes/ y.nbytes), x.shape, y.shape)

2)
for a,b,c in training_ds:
            print(c.shape, c.sum())

3)
a,b,c = training_ds[0]
"""