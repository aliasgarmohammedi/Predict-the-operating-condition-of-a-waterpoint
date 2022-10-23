import pickle
best_model_filename_xgb = 'finalized_model_xgboost.sav'
best_model_filename_rf = 'finalized_model.sav'
best_model_filename_dt = 'finalized_model_dt.sav'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import scipy as sp
import matplotlib as mpl
import seaborn as sns
import datetime

import warnings
warnings.filterwarnings("ignore")

def transform_feature(df, column_name):
    ''' Categorical data transformation based on unique values'''
    
    unique_values = set(df[column_name].tolist())
    transformer_dict = {}
    for index, value in enumerate(unique_values):
        transformer_dict[value] = index
    df[column_name] = df[column_name].apply(lambda y: transformer_dict[y])
    return df

def categorical_data_transformation(training):
    numerical_columns = ['days_since_recorded', 'population','gps_height','amount_tsh','longitude','latitude'] 
    columns_to_transform = [col for col in training.columns if col not in numerical_columns]
    for column in columns_to_transform: 
        training = transform_feature(training, column)
    return training

def normalize(df):
    '''Normalizes Column of a dataframe'''

    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def funder_cl(row):
    '''Keepig most frequent feature value for funder and assining other to non-frequent feature value'''
    
    if row['funder']=='Government Of Tanzania':
        return 'gov'
    elif row['funder']=='Danida':
        return 'danida'
    elif row['funder']=='Hesawa':
        return 'hesawa'
    elif row['funder']=='Rwssp':
        return 'rwssp'
    elif row['funder']=='World Bank':
        return 'world_bank'    
    elif row['funder']=='Kkkt':
        return 'Kkkt'
    elif row['funder']=='World Vision':
        return 'World Vision'
    elif row['funder']=='Unicef':
        return 'Unicef'
    elif row['funder']=='Tasaf':
        return 'Tasaf'
    elif row['funder']=='District Council':
        return 'District Council'
    else:
        return 'other'

def installer_cl(row):

    '''Keepig most frequent feature value for installer and assining other to non-frequent feature value'''
    
    if row['installer']=='DWE':
        return 'dwe'
    elif row['installer']=='Government':
        return 'gov'
    elif row['installer']=='RWE':
        return 'rwe'
    elif row['installer']=='Commu':
        return 'commu'
    elif row['installer']=='DANIDA':
        return 'danida'
    elif row['installer']=='KKKT':
        return 'kkkt'
    elif row['installer']=='Hesawa':
        return 'hesawa'
    elif row['installer']=='TCRS':
        return 'tcrs'
    elif row['installer']=='Central government':
        return 'Central government'
    else:
        return 'other'  

def scheme_cl(row):

    '''Keepig most frequent feature value for scheme_management and assining other to non-frequent feature value'''

    if row['scheme_management']=='VWC':
        return 'vwc'
    elif row['scheme_management']=='WUG':
        return 'wug'
    elif row['scheme_management']=='Water authority':
        return 'wtr_auth'
    elif row['scheme_management']=='WUA':
        return 'wua'
    elif row['scheme_management']=='Water Board':
        return 'wtr_brd'
    elif row['scheme_management']=='Parastatal':
        return 'Parastatal'
    elif row['scheme_management']=='Private operator':
        return 'pri_optr'
    elif row['scheme_management']=='SWC':
        return 'swc'
    elif row['scheme_management']=='Company':
        return 'company'
    elif row['scheme_management']=='Trust':
        return 'trust'
    else:
        return 'other'

def construction_cl(row):

    '''Converting the datatime into categorical as in : '60s', '70s', '80s', '90s, 
    '00s', '10s', 'unknown' which dosent have any year value for our convinience'''

    if row['construction_year'] >= 1960 and row['construction_year'] < 1970:
        return '60s'
    elif row['construction_year'] >= 1970 and row['construction_year'] < 1980:
        return '70s'
    elif row['construction_year'] >= 1980 and row['construction_year'] < 1990:
        return '80s'
    elif row['construction_year'] >= 1990 and row['construction_year'] < 2000:
        return '90s'
    elif row['construction_year'] >= 2000 and row['construction_year'] < 2010:
        return '00s'
    elif row['construction_year'] >= 2010:
        return '10s'
    else:
        return 'unknown'

def extraction_cl(row):

    '''Keepig most frequent feature value for extraction_type and assining other to non-frequent feature value'''

    if row['extraction_type']=='gravity':
        return 'gravity'
    elif row['extraction_type']=='nira/tanira':
        return 'nira'
    elif row['extraction_type']=='submersible':
        return 'submersible'
    elif row['extraction_type']=='swn 80':
        return 'swn'
    elif row['extraction_type']=='mono':
        return 'mono'
    elif row['extraction_type']=='india mark ii':
        return 'indiamark2'
    elif row['extraction_type']=='afridev':
        return 'afridev'
    elif row['extraction_type']=='ksb':
        return 'ksb'
    elif row['extraction_type']=='windmill':
        return 'windmill'
    elif row['extraction_type']=='india mark iii':
        return 'indiamark3'
    else:
        return 'other'

def Functional_status_prediction(model_filename,input):
    
    training_df = pd.read_csv('Training Set.csv')
    #train_labels = pd.read_csv('Training Set Labels.csv')
    #training_df = pd.merge(train_data, train_labels)
    
    training_df.loc[len(training_df)] = input
    
    training_df = training_df.drop(['id','source','wpt_name', 'num_private','district_code','region_code', 
          'quantity','quality_group','lga','ward','management', 'payment', 
           'extraction_type_group','extraction_type_class','waterpoint_type_group','recorded_by'],axis = 1)
    
    training_df['funder'] = training_df.apply(lambda row: funder_cl(row), axis=1)
    training_df['installer'] = training_df.apply(lambda row: installer_cl(row), axis=1)
    training_df.subvillage = training_df.subvillage.fillna('other')
    training_df.public_meeting = training_df.public_meeting.fillna('Unknown')
    training_df['scheme_management'] = training_df.apply(lambda row: scheme_cl(row), axis=1)
    training_df.scheme_name = training_df.scheme_name.fillna('other')
    training_df.permit = training_df.permit.fillna('Unknown')
    training_df.construction_year = pd.to_numeric(training_df.construction_year)
    training_df['construction_year'] = training_df.apply(lambda row: construction_cl(row), axis=1)
    training_df['extraction_type'] = training_df.apply(lambda row: extraction_cl(row), axis=1)
    training_df.date_recorded = pd.datetime(2013, 12, 3) - pd.to_datetime(training_df.date_recorded)
    training_df.columns = ['days_since_recorded' if x=='date_recorded' else x for x in training_df.columns]
    training_df.days_since_recorded = training_df.days_since_recorded.astype('timedelta64[D]').astype(int)
    training_df['longitude'] = training_df['longitude'].map( lambda x : training_df.longitude.mean() if x == 0 else x)
    training_df['latitude'] = training_df['latitude'].map( lambda x : training_df.latitude.mean() if x > -1 else x)

    #training_df = training_df.drop('status_group', axis=1)
    #training_df = training_df.drop('Unnamed: 0', axis=1)

    training = categorical_data_transformation(training_df)
    training = normalize(training)

    training_list = training.values.tolist()
    #newly added row extracting
    input = training_list[-1:]

    loaded_best_model = pickle.load(open(model_filename, 'rb'))

    sample_features = np.array(input)
    sample_features = sample_features.reshape(1, -1)

    predicted_label = loaded_best_model.predict(sample_features)

    if predicted_label[0] == 0:
      print('Pump is functional.')
    elif predicted_label[0] == 1:
      print('Pump is functional but needs repair.')
    else:
      print('Pump is non-functional.')

test = pd.read_csv('Test Set.csv')

test_list = test.values.tolist()
input_1 = test_list[78]

Functional_status_prediction(best_model_filename_dt,input_1)