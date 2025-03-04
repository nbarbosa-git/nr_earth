# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from joblib import dump, load
import pyarrow.feather as feather
import glob

def get_data(nrows=None, low_memory=False, dataset="training", feather=True): 



    #DOWNLOAD DATAFRAME
    if feather==True:
    	df = pd.read_feather('../../Data/Interim/'+dataset+'_val3.feather').iloc[:nrows,:]

    elif dataset == "legacy":
        data_path = '/content/legacy_compressed.feather' #colab
        df = pd.read_feather(data_path).iloc[:nrows,:]


    else:
    	data_path = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_'+dataset+'_data.csv'
    	df = pd.read_csv(data_path, nrows=nrows)





    #Low memory
    if low_memory == True:
        print("low memory activated")
        df = reduce_mem_usage(df, verbose=True)

    
    #COLUMN NAMES
    X = [c for c in df if c.startswith("feature")]
    y = "target"

    if dataset == "tournament": df.replace(to_replace='eraX', value='era9999', inplace=True)
    if (dataset != "legacy") and (dataset != "validation") and (feather==True): df['era'] = df.loc[:, 'era'].str[3:].astype('int32')
    if (dataset == "validation") and (feather==False): df['era'] = df.loc[:, 'era'].str[3:].astype('int32')
    if (dataset == "training") and (feather==False): df['era'] = df.loc[:, 'era'].str[3:].astype('int32')
    if dataset=='legacy': df['era'] = df['era'].astype('int32')


    #PRINT MEMORY USAGE
    print(df.info())
    return df, X, y




def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    #df.to_csv('../../data/interim/numerai_training_data_low_memory.csv')

    return df

#training_data = reduce_mem_usage(pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz"))
#training_data.head()


########################################################################################
########################################################################################
########################################################################################





def get_preditions(df, models, preds_type='legacy'):
  root_path = '/content/dissertacao/reports/predicoes_'+preds_type+'/'
  files_list = glob.glob(root_path+'**/*.csv', recursive = True)

  if preds_type=='legacy':
    ext = '_predictions.csv'


  if preds_type=='validacao':
    ext = '_preds_test.csv'

  if preds_type=='train':
    ext = '_preds_train.csv'

  for model in models:
    print(model)
    model_ = '/'+ model + ext
    csv_path = [s for s in files_list if model_ in s]
    print(csv_path)
    if len(csv_path) > 1: print('path: ', csv_path)
    if len(csv_path) < 1: print('model: ', model_)

    df[model] = pd.read_csv(csv_path[0], index_col='id').values.reshape(1,-1)[0]

  return df

def create_dtype():
	#download Numerai training data and load as a pandas dataframe
	TRAINING_DATAPATH = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
	df = pd.read_csv(TRAINING_DATAPATH, nrows=100) ## se der erro tira o nrows

	#create a list of the feature columns
	features = [c for c in df if c.startswith("feature")]

	#create a list of the column names
	col_list = ["id", "era", "data_type"]
	col_list = col_list + features + ["target"]

	#create a list of corresponding data types to match the column name list
	dtype_list_back = [np.float32] * 311
	dtype_list_front = [str, str, str]
	dtype_list = dtype_list_front + dtype_list_back

	#use Python's zip function to combine the column name list and the data type list
	dtype_zip = zip(col_list, dtype_list)

	#convert the combined list to a dictionary to conform to pandas convention
	dtype_dict = dict(dtype_zip)

	#save the dictionary as a joblib file for future use
	#dump(dtype_dict, 'dtype_dict.joblib')
	return dtype_dict


#####


def create_feather_df(dataset="training"):

	#load dictionary to import data in specific data types
	dtype_dict = create_dtype()


	#Get Data
	if dataset == "validation": FILE_URL  = '../../Data/Interim/'+dataset+'_data.csv'

	else: FILE_URL  = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_'+dataset+'_data.csv.xz'
	
	#file Name	
	FILE_NAME = '../../Data/Interim/' + dataset + '_compressed.feather'


	#download Numerai training data and load as a pandas dataframe
	df = pd.read_csv(FILE_URL, dtype=dtype_dict)

	#download Numerai tournament data and load as a pandas dataframe
	TOURNAMENT_DATAPATH = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'
	#df_tournament = pd.read_csv(TOURNAMENT_DATAPATH, dtype=dtype_dict)

	#save Numerai training data as a compressed feather file
	feather.write_feather(df, FILE_NAME)

	return df




###################


import numerapi
def get_massive_data_int8(dataset, download=True):

  file_path = "numerai_"+dataset+"_data_int8.parquet"

  if download==True:
    napi = numerapi.NumerAPI(verbosity="info")
    napi.download_dataset(file_path, "datasets/"+file_path)

  data = pd.read_parquet("datasets/"+file_path)
  return data

def get_massive_data_tv4(slice_data, download):

  
  training_data = get_massive_data_int8("training", download)
  validation_data = get_massive_data_int8("validation", download)

  mass_data_train_dict = dict()
  mass_data_val_dict = dict()
  mass_data = pd.concat([training_data,validation_data])

  for i in slice_data:
    mass_data_train_dict['train_'+str(i)] = mass_data[mass_data.era.isin(mass_data.era.unique()[i-1::8])]
    mass_data_val_dict['val_'+str(i)] = mass_data[mass_data.era.isin(mass_data.era.unique()[i-1+4::8])]



  X = [c for c in mass_data if c.startswith("feature")]
  yList = [c for c in mass_data if c.startswith("target")]
  y="target"

  return mass_data_train_dict, mass_data_val_dict, X, yList, y



def massive_data_csv_float16(**hparams):
  features = [c for c in validation_data if c.startswith("feature")]
  col_list = ["id", "era", "data_type"]
  col_list = col_list + features + ["target", "target_nomi_20", "target_nomi_60", "target_jerome_20", "target_jerome_60",
  "target_janet_20", "target_janet_60", "target_ben_20", "target_ben_60", "target_alan_20",
  "target_alan_60", "target_paul_20", "target_paul_60", "target_george_20", "target_george_60","target_william_20", 
  "target_william_60", "target_arthur_20", "target_arthur_60", "target_thomas_20", "target_thomas_60"]


  dtype_list_back = [np.float16] * 1071
  dtype_list_front = [str, np.float16, str]
  dtype_list = dtype_list_front + dtype_list_back
  dtype_zip = zip(col_list, dtype_list)
  dtype_dict = dict(dtype_zip)

  dump(dtype_dict, 'dtype_dict1_new.joblib')

  dtype_dict = load('dtype_dict1_new.joblib')
  df = pd.read_csv('numerai_training_data.csv', dtype=dtype_dict, nrows=None)
  return df





