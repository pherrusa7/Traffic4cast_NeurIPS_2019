import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import pairwise_distances
import datetime

###################################################### DATA PREPROCESSING PARAMETERS
MM, TIME_SLOT = 60, 5 # In minutes
SLOTS_PER_HOUR = int(MM/TIME_SLOT)
TOTAL_SLOTS = 24*SLOTS_PER_HOUR # Number of time-slots in a day

# ts_serie = 12+3

###################################################### PANDAS DEFINITION
def get_columns(regions, num_slots= TOTAL_SLOTS):
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

def get_day_info_pandas(path, regions):
    # load
    data = np.load(path)
    
    # create pandas object with correct types and columns
    columns, types, features = get_columns(regions)
    df = pd.DataFrame(data=data, columns=columns)
    df = df.astype(types) 
    return df, features

###################################################### PLOTS

def get_features(attribute, features, num_slots=TOTAL_SLOTS):
    feat = {'Speed': features[:num_slots], 
            'Volume': features[num_slots:num_slots*2], 
            'Direction': features[num_slots*2:num_slots*3]}
    return feat[attribute]
    
def get_dow(day):    
    date = datetime.datetime.strptime(str(day), '%Y%m%d')
    dow = ['day_of_week_'+str(i) for i in range(7)]
    return dow[date.weekday()]

def text_dow(day):
    date = datetime.datetime.strptime(str(day), '%Y%m%d')
    dow = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    return dow[date.weekday()]
    
def plot_day_dow_allDays(df, ax, col, features, key_feature = 'Speed', day=None,
                         bins=96, ec='#CC4F1B', fc='#089FFF', a=0.2, 
                         extra_info=False, show_std=True, num_slots=TOTAL_SLOTS):
    
    # get features to plot
    features = get_features(key_feature, features)
    text_agg = "all days"

    # 1.1 plot the mean for all days
    if extra_info:
        count = df.shape[0]
        df[features].mean().plot(label="Agg by "+text_agg+" ("+str(count)+' days)', ax=ax, color='b', alpha=a, style='-')

    
    if day is not None:
        # 1.2 plot the day
        # I need to to this since (pandas-dev/pandas/issues/9542, /pandas-dev/pandas/issues/10119)
        pd.Series((df[df.day == day][features].values.squeeze()), name=str(day)+' ('+ text_dow(day) +')').plot(ax=ax, color=col, style='*-', legend=True)
#         pd.Series((df[df.day == day][get_features('Volume')].values.squeeze()), name=str(day)+' ('+ text_dow(day) +')').plot(ax=ax, color=col, style='*-', legend=True)
#         pd.Series((df[df.day == day][get_features('Direction')].values.squeeze()), name=str(day)+' ('+ text_dow(day) +')').plot(ax=ax, color=col, style='*-', legend=True)

        # before I was using the line below, but label was not shown, only the index of the dataframe instead
        #df[df.day == day][features[:num_slots]].T.plot(figsize=(25,8), label=str(day), ax=ax, color=col, style='*-')
    
        # 1.3 plot only the same day of the week days
        df = df[df[get_dow(day)] == 1]
        text_agg = text_dow(day)
        
    count = df.shape[0]
    df = df[features].describe()

    df.loc['mean'].T.plot(ax=ax, label="Agg by "+text_agg+" ("+str(count)+' days)', color='blue', alpha=a, style='--')

    # 4. set standard deviation as shaded
    if show_std:
        plt.fill_between(range(num_slots), df.loc['mean'] + df.loc['std'] , df.loc['mean'] - df.loc['std'], alpha=a, edgecolor=ec, facecolor=fc, 
            label="+- std on all "+text_agg+" ("+str(count)+' days)')

def time_slots_function(function=None):
    """ creates a sorted list of dates from 00:00:00 to 24:00:00 """
    l = []
    for i in np.arange(0, 24, 1):
        for j in np.arange(0, 60, 15):
            if i<10:
                s_i = '0'+str(i)
            else:
                s_i = str(i)
            if j<10:
                s_j = '0'+str(j)
            else:
                s_j = str(j)
            if function:
                s = s_i+s_j+'00'
            else:
                s = s_i+':'+s_j
            l.append(s)
    l.append(l[0])
    return l

def plot_traffic_pattern(df, city, features, day=None, attribute='Speed', info_all_days=False, show_std=True, num_slots=TOTAL_SLOTS, event=None):
    """ Use this to inspect your event pattern
    """
    # init plot
    fig = plt.figure(figsize=(25,8))
    ax = plt.subplot(111)
    
    # setting the title of the plot
    title = attribute
    title += ' - ' + city        
    if day is not None:
        title += ' - ' + str(day)

    # plot
    df = df[df.city==city]
    plot_day_dow_allDays(df, ax, 'C1', features, attribute, day, show_std=show_std, ec='#3F7F4C', fc='#7EFF99', a=0.3, extra_info=info_all_days)
    
    # info about the x-axis
    l = time_slots_function()
    slots_mapping = [l[i]+'-'+l[i+1] for i in range(len(l)-1)]
    
    if event is not None:
        # plot vertical lines at the start and end of a match
        plt.axvline(event['start'], color='black')
        plt.axvline(event['end_game'], color='black')

    # setting the plot
    plt.title(title, fontsize = 20)
    plt.legend(prop={'size': 15})
    plt.xticks([i for i in range(1, num_slots+1) if i%3==0], slots_mapping, rotation=45, ha='right')
    plt.show()

def plot_traffic_patterns_w_event(df, city, features, day=None, info_all_days=False, show_std=True, sub_set='training', event=None):
    if sub_set in ['training', 'validation']:
        df = df[df.subset.isin(['training', 'validation'])]
        print("Plots based on data in:", ['training', 'validation'])
    else:
        print("Plots based on data in:", ['training', 'validation', 'test'])
    
    for attribute in ['Speed', 'Volume', 'Direction']:
        print("--> {}:".format(attribute))
        plot_traffic_pattern(df, city, features, day, attribute, info_all_days, show_std, event=event)
            
def plot_traffic_patterns(df, features, day=None, info_all_days=False, show_std=True, only_train_val=True):
    if only_train_val:
        df = df[df.subset.isin(['training', 'validation'])]
        print("Plots based on data in:", ['training', 'validation'])
    else:
        print("Plots based on data in:", ['training', 'validation', 'test'])
    
    for city in df.city.unique():
        print("\n\n========== {} ==========".format(city))
        for attribute in ['Speed', 'Volume', 'Direction']:
            print("--> {}:".format(attribute))
            plot_traffic_pattern(df, city, features, day, attribute, info_all_days, show_std)