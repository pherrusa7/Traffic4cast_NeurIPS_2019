
import numpy as np
import pandas as pd

DAYS_INFO_PATH, REGIONS = 'agg_data/1_regions_features_mean.npy', 1 # data aggregated for the whole city

###################################################### DATA PREPROCESSING PARAMETERS
MM, TIME_SLOT = 60, 5 # In minutes
SLOTS_PER_HOUR = int(MM/TIME_SLOT)
TOTAL_SLOTS = 24*SLOTS_PER_HOUR # Number of time-slots in a day

WEATHER_CONTINOUS=['tempC', 'FeelsLikeC', 'windspeedKmph', 'precipMM', 'visibilityKm'] 
WEATHER_CATEGORICAL=['weatherDesc', 'weatDesc_Id'] 
WEATHER_CATEGORICAL_TARGET=['weather_Cloudy', 'weather_Fog', 'weather_Heavy rain', 'weather_Heavy snow', 'weather_Light drizzle', 
                            'weather_Light freezing rain', 'weather_Light rain', 'weather_Light rain shower', 'weather_Light sleet', 
                            'weather_Light sleet showers', 'weather_Light snow', 'weather_Light snow showers', 'weather_Mist', 
                            'weather_Moderate or heavy freezing rain', 'weather_Moderate or heavy rain shower', 
                            'weather_Moderate or heavy snow showers', 'weather_Moderate rain', 'weather_Moderate rain at times', 
                            'weather_Moderate snow', 'weather_Overcast', 'weather_Partly cloudy', 'weather_Patchy light drizzle', 
                            'weather_Patchy light rain', 'weather_Patchy light rain with thunder', 'weather_Patchy light snow', 
                            'weather_Patchy rain possible', 'weather_Sunny', 'weather_Thundery outbreaks possible'] # 28

def get_hour_lookup():
    """ returns a dict mapping time_slots/time_bins to hours as string """
    hours = ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00',
             '06:00:00', '07:00:00', '08:00:00', '09:00:00', '10:00:00', '11:00:00',
             '12:00:00', '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00', 
             '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00']

    tod_lookup_hourly = {}

    i = -1
    for j in range(288):
        if j%12==0:
            i += 1
        tod_lookup_hourly[j] = hours[i]

    return tod_lookup_hourly

###################################################### PANDAS DEFINITION
def get_columns(regions, num_slots= TOTAL_SLOTS):
    """
        define features for a dataframe containing time-flow info
    """
    ## define columns
    columns = ['city', 'subset', 'day'] + ['day_of_week_'+str(i) for i in range(7)] + ['day_num_'+str(i) for i in range(1, 32)] + ['month_'+str(i) for i in range(1, 13)]
    types   = [np.str, np.str, np.int] + [np.int for i in range(7)] + [np.int for i in range(1, 32)] + [np.int for i in range(1, 13)]
    features = []
    for j in range(regions**2):
        tmp = ''
        if regions > 1:
            tmp = 'r'+str(j+1)+str(regions**2)+'_'
            
        features += [tmp+'slot_speed_'+str(i) for i in range(num_slots)] + \
                   [tmp+'slot_volume_'+str(i) for i in range(num_slots)] + \
                   [tmp+'slot_direction_'+str(i) for i in range(num_slots)] 
        types   += [np.float for i in range(num_slots)] + \
                   [np.float for i in range(num_slots)] + \
                   [np.float for i in range(num_slots)] 
    columns += features
    
    types_dict = {}
    for i, col in enumerate(columns):
        types_dict[col] = types[i]
    return columns, types_dict, features

def get_pandas(path, regions):
    """ simply load file as pandas and returns also its features (columns) """
    # load
    data = np.load(path) # contains aggregated info for all cities in all time-slots for speed, volume & heading
    
    # create pandas object with correct types and columns
    columns, types, features = get_columns(regions)
    df = pd.DataFrame(data=data, columns=columns)
    df = df.astype(types) 
    return df, features

###################################################### INFORMATION ABOUT TIME-FLOW
def get_time_vector(city, subset, days_info_path=DAYS_INFO_PATH, regions=REGIONS):
    """
        Get a dict with time-flow info from a dataframe
        returns: dict with 'dates' as keys
    """
    att = ['city', 'subset', 'day', 
           # day of week (dow)
           'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 
           # day of mont (dom)
           'day_num_1', 'day_num_2', 'day_num_3', 'day_num_4', 'day_num_5', 'day_num_6', 'day_num_7', 'day_num_8', 'day_num_9', 'day_num_10', 'day_num_11', 
           'day_num_12', 'day_num_13', 'day_num_14', 'day_num_15', 'day_num_16', 'day_num_17', 'day_num_18', 'day_num_19', 'day_num_20', 'day_num_21', 
           'day_num_22', 'day_num_23', 'day_num_24', 'day_num_25', 'day_num_26', 'day_num_27', 'day_num_28', 'day_num_29', 'day_num_30', 'day_num_31', 
           # month of the year (moy)
           'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']
    
    # create df
    df, features = get_pandas(days_info_path, regions)
    
    # select subset and city
    df = df[df.subset == subset]
    if city != 'all':
        df = df[df.city == city]
        
    # select attributes
    df = df[att[2:]]
    
    # create a dict to easy access them when creating batches
    values = {}
    for day in df.get_values():
        values[str(day[0])] = day[1:]
        
    return values

def get_time_slot_continous(time_slot, max_v = 288):
    """ example of return
         x: [0.00545413 0.99998513] 
         x.shape:(2,) 
         type(x): <class 'numpy.ndarray'> """
    s = np.sin(((1./2)*np.pi*time_slot)/max_v)
    c = np.cos(((1./2)*np.pi*time_slot)/max_v)
    return np.array([s, c])

def get_frame_info(time_dict, date, time_slot):
    """
        Load time-flow info for a frame
    """
    # info of day_of_week, day_of_month, and month_of_year
    dow_dom_moy = time_dict[date]
    
    time_slot_encoding = get_time_slot_continous(time_slot)
    
    return dow_dom_moy, time_slot_encoding

def get_sequence_time_info(path, idx, length, time_dict): # not used
    """
        Load time-flow info for each frame in sequence
    """
    # get date
    date = path.split("/")[-1].split("_")[0]
    
    # get info for frame in sequence
    seq_day_info, seq_time_info = [], []
    for time_slot in range(idx, idx + length):
        day_info, time_info = get_frame_info(time_dict, date, time_slot)
        
        seq_day_info.append(day_info)
        seq_time_info.append(time_info)

    seq_day_info = np.asarray(seq_day_info)
    seq_time_info = np.asarray(seq_time_info)
    return seq_day_info, seq_time_info

def reframe_including_future_known_info(x, len_seq_out = 3): # not used
    """
        Reframe data to include current information + future information of 'len_seq_out' steps more
        Input: x.shape (6, 50)
        outpout x.shape (3, 50 + 50*len_seq_out) e.g., = (3, 50 + 50*3) = (3, 200)
    """
    sequence_with_future_info = []

    # for each input frame in the sequence
    for i in range(x.shape[0]-len_seq_out):

        x_i = []
        # load the current frame info + 'len_seq_out=3' steps more
        for next_frame in range(len_seq_out + 1):
    #         print("Adding init {} and {}, then {}".format(i, next_frame, i + next_frame))
            x_i.extend(x[i + next_frame])
        sequence_with_future_info.append(x_i)
#     print("")
    return np.asarray(sequence_with_future_info)

def reframe_with_future(x, t):
    x = reframe_including_future_known_info(x)
    t = reframe_including_future_known_info(t)
    return x, t
    
def get_batch_day_time_info(batch, seq_length, time_dict): # not used
    """
        Load time-flow info for each sequence in batch
    """
    batch_day_info, batch_time_info = [], []

    for path, init_seq_time_slot in batch:
        # get info for the whole seq in current sample
        seq_day_info, seq_time_info = get_sequence_time_info(path, init_seq_time_slot, seq_length, time_dict) 
        
        # reframe sequence to match input shape containing info about the known future
        seq_day_info, seq_time_info = reframe_with_future(seq_day_info, seq_time_info)
        
        batch_day_info.append(seq_day_info)
        batch_time_info.append(seq_time_info)

    batch_day_info, batch_time_info = np.asarray(batch_day_info), np.asarray(batch_time_info)
    return batch_day_info, batch_time_info

###################################################### INFORMATION ABOUT WEATHER
def normalize(df, columns):
    result = df.copy()
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def get_target_df(df, att_continous=WEATHER_CONTINOUS, att_categorical=WEATHER_CATEGORICAL):
    # create for weather descriptions
    dict_weather = {}
    for i, c in enumerate(df.groupby('weatherDesc')['weatherDesc']):
        dict_weather[c[0]] = i

    # create a new column based on that
    df['weatDesc_Id'] = df['weatherDesc'].map(lambda x: dict_weather[x])

    # limit the dataframe to target variables
    df = df[att_continous+att_categorical]
    
    # normalize numerical columns
    df = normalize(df, att_continous)
    
    # create dummy vars for categorical var 'WeatherDesc'
    df = pd.concat([df, pd.get_dummies(df['weatherDesc'], prefix='weather', drop_first=True)],axis=1)
    
    return df

def get_weather_df(city):
    """
        Get a dataframe of weather forecastings
    """
    root = '/home/pherruzo/data/events/Weather/'
    
    # load df on target vars
    df = pd.read_csv(root + city + '_Weather.csv', index_col=0)
    df = get_target_df(df)
    
    return df

def get_frame_weather_info(df, date, time_slot, att_continous=WEATHER_CONTINOUS, att_categorical=WEATHER_CATEGORICAL_TARGET):
    """
        Load weather info for a frame
    """
    # get time of day from time_slot
    tod_lookup_hourly = get_hour_lookup()    
    tod = tod_lookup_hourly[time_slot]
    
    # transform date to correct format
    date = '20180606'
    date = date[:4]+'-'+date[4:-2]+'-'+date[-2:]
    
    # get categorical & numerical data
    cont_values = df.loc[date+' '+tod][att_continous].get_values()
    categorical_values = df.loc[date+' '+tod][att_categorical].get_values() # [-1:] # only when using sparse weatherDesc
    
    return categorical_values, cont_values

###################################################### GENERAL LOADERS FOR EXOGENOUS DATA

def get_sequence_info_general(path, idx, length, data_holder, get_frame_func):
    """
        Load exogenous info for each frame in sequence, info depends on function 'get_frame_values'
    """
    # get date
    date = path.split("/")[-1].split("_")[0]
    
    # get info for frame in sequence
    categorical_values, cont_values = [], []
    for time_slot in range(idx, idx + length):
        categorical_val, cont_val = get_frame_func(data_holder, date, time_slot)
        
        categorical_values.append(categorical_val)
        cont_values.append(cont_val)

    categorical_values = np.asarray(categorical_values)
    cont_values = np.asarray(cont_values)
    return categorical_values, cont_values

def get_seq_exogenous(path, init_seq_time_slot, seq_length, data_holder, get_frame_func):
    """
        prepare sequence
    """
    
    # get info for the whole seq in current sample             
    seq_categorical, seq_continous = get_sequence_info_general(path, init_seq_time_slot, seq_length, data_holder, get_frame_func) 

    # reframe sequence to match input shape containing info about the known future
    seq_categorical, seq_continous = reframe_with_future(seq_categorical, seq_continous)
    
    return seq_categorical, seq_continous

def get_batch_exogenous_info(batch, seq_length, offset_if_test=0, time_dict=None, df_weather=None, df_soccer=None):
    """
        Load exogenous info for each sequence in batch
    """
    batch_day_info, batch_time_info = [], []
    batch_weather_categorical, batch_weather_continous = [], []

    for path, init_seq_time_slot in batch:
        # update init timeslot with the offset (0 always but in test)
        init_seq_time_slot += offset_if_test
        
        ############ TIME-FLOW INFO
        if time_dict is not None:
            categorical_seq, continous_seq = get_seq_exogenous(path, init_seq_time_slot, seq_length, time_dict, get_frame_info)

            batch_day_info.append(categorical_seq)
            batch_time_info.append(continous_seq)
        
        ############ WEATHER INFO
        if df_weather is not None:
            categorical_seq, continous_seq = get_seq_exogenous(path, init_seq_time_slot, seq_length, df_weather, get_frame_weather_info)

            batch_weather_categorical.append(categorical_seq)
            batch_weather_continous.append(continous_seq)
            
    # to numpy
    batch_day_info, batch_time_info = np.asarray(batch_day_info), np.asarray(batch_time_info)
    batch_weather_categorical, batch_weather_continous = np.asarray(batch_weather_categorical), np.asarray(batch_weather_continous)
    
    extra_vars = {}
    extra_vars['day_info'] = batch_day_info
    extra_vars['time_info'] = batch_time_info
    extra_vars['weather_categorical'] = batch_weather_categorical
    extra_vars['weather_continous'] = batch_weather_continous
    
    return extra_vars