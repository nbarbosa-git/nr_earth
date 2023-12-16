import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import seaborn as sns
import glob
import warnings
warnings.filterwarnings("ignore")


try:
  from src.validation import metrics
  from src.visualization import visualize
  
except:
  #import dsr
  import metrics_description
  import metrics
  import visualize







class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='columns')
        return json.JSONEncoder.default(self, obj)



def create_json(preds_path, model_metrics, feat_corrs, tb200_exp, era_scores, mmc_scores, tb200_scores, model_name, path=''):

  with open(path+model_name+'.json', 'w') as fp:
      json.dump({'preds_path': preds_path,
                'metrics':model_metrics,
                'feat_corrs':feat_corrs,
                'tb200_exp':tb200_exp,
                'era_scores':era_scores,
                'mmc_scores':mmc_scores,
                'tb200_scores':tb200_scores
                }, fp, cls=JSONEncoder, indent=4)
    
  return


def get_json_paths(data_type, local=False):

  if local==True: 
      root_path = '/Users/nicholasrichers/Documents/GitHub/dissertacao/reports/'
  else: 
      root_path = '/content/dissertacao/reports/'

  json_root_folder = root_path + data_type
  
  files_list = glob.glob(json_root_folder+'**/*.json', recursive = True) 
  return files_list


def check_json(json_file, model):
   
  json_dict = pd.read_json(json_file['metrics'], orient='index')
  name = json_dict["Model_Name"][0]

  if name != model: 
    print('ERRO')
    print('Deveria Pegar: ', model)
    print('Pegou: ', name)

  #else: print('sucesso')

  return



def get_json(model, data_type, json_files_list, local=False):


  #Get path
  #if local==True: root_path = '/Users/nicholasrichers/Documents/GitHub/dissertacao/reports/'
  #else: root_path = '/content/dissertacao/reports/' 

  try:
    json_file = json.load(open('/content/'+model+'.json'))


  except:
    model_ = '/'+ model + '.json'
    json_path = [s for s in json_files_list if model_ in s]
    #full_path = root_path + data_type + json_files_dict[model] + '.json'
    if len(json_path) > 1: print('path: ',json_path)
    if len(json_path) < 1: print('model: ', model_)
    json_file = json.load(open(json_path[0]))

  #check name
  check_json(json_file, model)

    

  return json_file



def get_scores(models, data_type, scores_type, local=False, **kwargs):

  #1 Get Json Files
  json_files = dict()
  json_files_list = get_json_paths(data_type, local)
  for model in models:
    json_files[model] = get_json(model, data_type, json_files_list, local)


  #1Get preds DF
  if (scores_type=='preds'):
    preds_path = dict()
    preds_df = pd.DataFrame([])

    for model in models:
      preds_path[model] = json_files[model]['preds_path']

      if local==True: 
        root_path = '/Users/nicholasrichers/Documents/GitHub/'
        preds_path[model] = root_path + preds_path[model][9:]

      try:
        preds_df[model] = pd.read_csv(preds_path[model], index_col='id').values.reshape(1,-1)[0]

      except: 
        print('Arquivo Não encontrado')
        print(preds_path[model])
    
    return preds_df



  #2 get era score or MMC scores DF
  if (scores_type=='era_scores') or (scores_type=='mmc_scores') or (scores_type=='tb200_scores'):
    scores_dict = dict()
    for model in models:
      try:
        scores = pd.read_json(json_files[model][scores_type], orient='index')
      except:
        scores = pd.read_json(json_files[model]['mmc_scores'], orient='index')


      scores_dict[model] = pd.Series(scores.values.reshape(1,-1)[0], index=scores.index)
    df_era_scores = pd.DataFrame.from_dict(scores_dict)
    return df_era_scores

  #3 get metrics DF
  if (scores_type=='metrics'):
    metrics_dict = dict()
    for model in models:
      metrics_dict[model] = pd.read_json(json_files[model][scores_type])
    df_metrics_cons = metrics.metrics_consolidated(metrics_dict)
    return df_metrics_cons

  #4 Feat corr dict of DF
  if (scores_type=='feat_corrs') or (scores_type=='tb200_exp'):
    corrs_dict_df = dict()
    for model in models:
      try:
        corrs_dict_df[model] = pd.read_json(json_files[model][scores_type])

      except:
        corrs_dict_df[model] = pd.read_json(json_files[model]['feat_corrs'])

    return corrs_dict_df


  else: 
    print('Score indisponível')
    print("Scores disponíveis:\n\
    preds \n\
    metrics \n\
    era_scores \n\
    mmc_scores \n\
    feat_corrs \n")
  return



########################################################################
########################################################################
########################################################################

#Metrics
def get_diagnostics(df_metrics, **hparams):

  leaderboard = df_metrics[df_metrics.Categoria.isin(hparams['category'])].loc[:,df_metrics.columns[:-3]]
  check = all(item in ["Performance", "Risk", "MMC"] for item in hparams['category'])

  #with colors
  if check: return leaderboard.astype(float).style.apply(visualize.diagnostic_colors).apply(hparams['highlight'], axis = 1)

  #No colors
  else: return leaderboard.astype(float).style.apply(hparams['highlight'], axis = 1)


#Era/MMC/TB Scores
def visualize_bars_scores(df_scores):

  if isinstance(df_scores, dict):
    models = list(df_scores['era_scores'].columns)
    temp_df = pd.DataFrame([])
    for model in models:
      #temp_df = pd.DataFrame([])
      for scores_type in ['era_scores', 'mmc_scores', 'tb200_scores']:
        temp_df[str(model)+"_"+scores_type] = df_scores[scores_type][model]
    df_scores = temp_df


  visualize.plot_era_scores(df_scores, df_scores.columns[:])
  return df_scores



def create_corr_fexp_df(df_scores, **hparams): 


  if df_scores['era_scores'].shape[0]==120:
    data_type='train'
  else: 
    data_type='legacy'

  path = 'https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/scores_feat_raw/era_scores_feat_raw_'
  era_scores_df = pd.read_csv(path+data_type+'.csv')
  
  if isinstance(df_scores, dict):
    models = list(df_scores['era_scores'].columns)
    temp_df = pd.DataFrame([])
    for model in models:
      #temp_df = pd.DataFrame([])
      for scores_type in ['era_scores', 'mmc_scores', 'tb200_scores']:
        temp_df[str(model)+"_"+scores_type] = df_scores[scores_type][model]



      for feature in hparams['features']:
        if feature.startswith("minus_"): 
          operator = -1
          feature_ = feature[6:]
        else: 
          operator = 1
          feature_ = feature

        for scores_type in ['feat_corrs', 'tb200_exp']:
          temp_df[str(feature)+"_"+scores_type] = operator * df_scores[scores_type][model][feature_]

        temp_df[str(feature)+"_"+'diff_exp'] = temp_df[str(feature)+"_"+'tb200_exp'] - temp_df[str(feature)+"_"+'feat_corrs']
        temp_df[str(feature)+"_"+'era_score'] = operator * era_scores_df[feature_]

    scores = temp_df


  return scores


def corr_fexp_eras(df_scores, **hparams):
  scores_rep = create_corr_fexp_df(df_scores, **hparams)
  gap = hparams['gap']
  corr_df_dict = dict()
  for eras in scores_rep.index[::gap]:
    corr_df_dict['eras_'+str(eras)+'_'+str(eras+gap-1)] = scores_rep.iloc[eras-1:eras+gap-1,:].corr(method=hparams['method']).iloc[3:,:3]
  corr_df_dict['all'] = scores_rep.corr(method=hparams['method']).iloc[3:,:3]


  eras_score = pd.DataFrame(columns=corr_df_dict.keys(), index=corr_df_dict['all'].index)
  #mmc_score = pd.DataFrame(columns=corr_df_dict.keys(), index=corr_df_dict['all'].index)
  #tb200_score = pd.DataFrame(columns=corr_df_dict.keys(), index=corr_df_dict['all'].index)

  for eras,corrs  in corr_df_dict.items():
    eras_score[eras] = corrs.take([hparams['score_type']], axis=1)
    #mmc_score[eras] = corrs.take([1], axis=1)
    #tb200_score[eras] = corrs.take([2], axis=1)


  f, ax = plt.subplots(figsize=(10,2)) #figure(figsize=(19.20,10.80))


  # Generate a custom diverging colormap
  v= hparams['v']
  cmap = sns.diverging_palette(230, 20, as_cmap=True)
  #sns.heatmap(feat_scores.loc[:,feat_].T, cmap=cmap, center=0, vmin=-v, vmax=v)
  sns.heatmap(eras_score.loc[:,:], cmap=cmap, center=v-.2, vmin=v-.4, vmax=v,
                                           linewidths=.75, cbar_kws={"shrink": .8},  annot=True)


  return eras_score#,mmc_score,tb200_score



#Era/MMC/TB Scores
def visualize_bars_scores_full(df_scores, **hparams):

  scores = create_corr_fexp_df(df_scores, **hparams)
  visualize.plot_era_scores(scores, scores.columns[:])
  return scores





def correlation_scores(df_scores):
  visualize.plot_corr_matrix_full(df_scores)
  return df_scores

def get_scores_models(df_scores):
  #visualize.plot_corr_matrix_full(df_scores)
  return df_scores

def scores_correlation(scores_dict, **hparams):

  models = list(scores_dict['era_scores'].columns)
  flat_corr_matrix = pd.DataFrame(index=['era_mmc', 'era_tb', 'mmc_tb'])

  for model in models:
    temp_df = pd.DataFrame([])
    for scores_type in ['era_scores', 'mmc_scores', 'tb200_scores']:
      temp_df[scores_type] = scores_dict[scores_type][model]
    flat_matrix = temp_df.corr().values.flatten()
    flat_corr_matrix[model] = [flat_matrix[1]]+[flat_matrix[2]]+[flat_matrix[5]] #3x3 Matrix
    #print(temp_df.corr())


  f, ax = plt.subplots(figsize=(10,10)) #figure(figsize=(19.20,10.80))
  #print(mod_)

  # Generate a custom diverging colormap
  v=hparams['v']
  cmap = sns.diverging_palette(230, 20, as_cmap=True)
  #sns.heatmap(feat_scores.loc[:,feat_].T, cmap=cmap, center=0, vmin=-v, vmax=v)
  sns.heatmap(flat_corr_matrix.loc[:,:].T, cmap=cmap, center=v-.2, vmin=v-.4, vmax=v,
                                           linewidths=.75, cbar_kws={"shrink": .8},  annot=True)

  #plt.savefig('myimage.png', format='png', dpi=1200)
  plt.show()

  return flat_corr_matrix.T


def clustermap_scores(df_scores):
  cmap = sns.diverging_palette(220, 10, as_cmap=True)
  g = sns.clustermap(df_scores.corr(method="spearman"), cmap=cmap, annot=True)
  return df_scores

#preds
def get_preds(preds, **hparams):
  #preds['id'] = hparams['df'].id
  return preds


#feat corrs
def plot_exposures_map(corrs_dict_df, **hparams):
  feat_ = []

  #for fg in hparams['groups']: feat_ += feature_groups[fg]


  for mod_, feat_scores in corrs_dict_df.items():   
    # Set up the matplotlib figure  ##target
    f, ax = plt.subplots(figsize=(19.20,10.80)) #figure(figsize=(19.20,10.80))
    print(mod_)

    # Generate a custom diverging colormap
    v=hparams['v']
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    #sns.heatmap(feat_scores.loc[:,feat_].T, cmap=cmap, center=0, vmin=-v, vmax=v)
    sns.heatmap(feat_scores.loc[:,:].T, cmap=cmap, center=0, vmin=-v, vmax=v)

    #plt.savefig('myimage.png', format='png', dpi=1200)
    plt.show()



def create_report(models, data_type, local=False, **report):

  #get scores
  if (report['scores_type']=='all_scores') or (report['scores_type']=='all_scores_exp'):
    scores_dict = dict()
    for scores_type in ['era_scores', 'mmc_scores', 'tb200_scores']:
      scores_dict[scores_type] = get_scores(models, data_type, scores_type, local)

    #get exposures map
    if report['scores_type']=='all_scores_exp':
      for scores_type in ['feat_corrs', 'tb200_exp']:
        scores_dict[scores_type] = get_scores(models, data_type, scores_type, local)

    rep = report['func'](scores_dict, **report['hparams'])
    scores = rep


  elif (report['scores_type']=='corr_mmc'):
    scores = get_scores(models, data_type, 'era_scores', local) + 1*get_scores(models, data_type, 'mmc_scores', local)
    rep = report['func'](scores, **report['hparams'])


  elif (report['scores_type']=='corr_2mmc'):
    scores = get_scores(models, data_type, 'era_scores', local) + 2*get_scores(models, data_type, 'mmc_scores', local)
    rep = report['func'](scores, **report['hparams'])



  else:
    scores = get_scores(models, data_type, report['scores_type'], local)
    rep = report['func'](scores, **report['hparams'])

  if report['scores_type']=='metrics' : scores = rep
  return  scores



########################################################################
########################################################################
########################################################################




report_args = {
    
    #Main Diagnostics
    'Diagnostics': {'scores_type':'metrics', 
                     'func': get_diagnostics, 
                     'hparams': {'category':["Performance", "Risk", "MMC"], 
                                 'highlight': visualize.highlight_max}}, 


    #Alt. Diagnostics ['MMC_FN', 'Financeira', 'Estatistica', 'AR', 'Live', 'Special']
    'Diagnostics2': {'scores_type':'metrics', 
                     'func': get_diagnostics, 
                     'hparams': {'category':["TB", "Financeira", "Special"], 
                                 'highlight': visualize.highlight_max}}, #highlight_top3

    #era/mmc_scores
    'Scores_Viz': {'scores_type':'era_scores', #mmc_scores era_scores tb200_scores
                     'func': visualize_bars_scores, 
                     'hparams': {}},
               

    #era/tb/mmc_scores
    'Scores_Corr': {'scores_type':'era_scores', #mmc_scores era_scores tb200_scores
                     'func': correlation_scores, 
                     'hparams': {}},


    #era/tb/mmc_scores
    'Get_Scores': {'scores_type':'era_scores', #mmc_scores era_scores tb200_scores
                     'func': get_scores_models, 
                     'hparams': {}},
               
    #era/tb/mmc_scores
    'Scores_Cluster': {'scores_type':'era_scores', #mmc_scores era_scores 
                     'func': clustermap_scores, 
                     'hparams': {}},


    #era/tb/mmc_scores
    'Corr_Scores': {'scores_type':'all_scores', 
                     'func': scores_correlation, 
                     'hparams': {'v': .8}},

    #preds
    'Preds': {'scores_type':'preds', 
                     'func': get_preds, 
                     'hparams': {'df': None}}, #df_validation
               
    
    'Exp_Map': {'scores_type':'feat_corrs', #tb200_exp
                     'func': plot_exposures_map, 
                     'hparams': {'v': .15,
                                 'groups': ["intelligence", #NOT WORKING
                                            "wisdom", 
                                            "charisma", 
                                            "dexterity", 
                                            "strength",  
                                            "constitution"
                                            ]}},


    #era/mmc_scores
    'Fexp_Scores_Viz': {'scores_type':'all_scores_exp', #mmc_scores era_scores tb200_scores
                     'func': visualize_bars_scores_full, 
                     'hparams': {'features': ['feature_dexterity7']}}, #Fexp



     'Fexp_Scores_Corr': {'scores_type':'all_scores_exp', 
                     'func': corr_fexp_eras, 
                     'hparams': {'features': ['feature_dexterity7'],
                                 'v': .4,
                                 'gap': 10,
                                 'score_type': 0}},#era:0, mmc:1, tb200:2



}


#models = ['nr_shanghai', 'nr_singapore']
#data_type = 'json_legacy/'
#report = 'Exp_Map'


#_ = create_report(models, data_type, local=False, **report_args[report])

#reports_json.create_report(models, data_type, local=True, **reports_json.report_args['Diagnostics'])
#reports_json.create_report(models, data_type, local=True, **reports_json.report_args['Diagnostics2'])
#era_scores = reports_json.create_report(models, data_type, local=True, **reports_json.report_args['Scores_Viz'])
#era_scores=reports_json.create_report(models, data_type, local=True, **reports_json.report_args['Scores_Corr'])
#era_scores=reports_json.create_report(models,data_type,local=True, **reports_json.report_args['Scores_Cluster'])
#preds=reports_json.create_report(models,data_type,local=True, **reports_json.report_args['Preds'])
#exp_map=reports_json.create_report(models,data_type,local=True, **reports_json.report_args['Exp_Map'])
#era_scores=reports_json.create_report(models, data_type, local=True, **reports_json.report_args['Corr_Scores'])
#era_scores=reports_json.create_report(models, data_type, local=True, **reports_json.report_args['Fexp_Scores_Viz'])
#era_scores=reports_json.create_report(models, data_type, local=True, **reports_json.report_args['Fexp_Scores_Corr'])



