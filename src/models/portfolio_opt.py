##########################################################################################
##########################################################################################
##########################################################################################



try:
    from src.models import fs_dynamic_synthetic
except:
    import fs_dynamic_synthetic


def get_dom_preds(era_col, model_col, fn_model_col, dom):
  
  era = era_col.unique()[0]
  dom_era = dom[era]

  if dom_era==1:
    return model_col

  else:
    return fn_model_col



def get_model_dominance_preds(ddf, models, data_type, dom_sign, dom_source):

  df = ddf.copy()
  
  #legacy only
  root_path = 'https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/predicoes_legacy/raw/'

  for model_name in models: #['gbmsnr_15', 'fn_gbmsnr_15']
    model_path = root_path+model_name+'_predictions.csv'
    df[model_name] =  pd.read_csv(model_path, index_col='id').values.reshape(1,-1)[0]

  dom = fs_dynamic_synthetic.get_dominance_eras(data_type, dom_sign, dom_source) #dummy or not
  preds_dom = df.groupby('era').apply(lambda x: 
                                 get_dom_preds(x['era'], x[models[0]], x[models[1]], dom)
                                 ).values

  return preds_dom



#df = df_validation.copy()
#models = ['gbmsnr_15', 'fn_gbmsnr_15']
#data_type='legacy' 
#dom_sign=1
#dom_source='git'
#
#
#preds = get_model_dominance_preds(df_validation, models, data_type, dom_sign, dom_source)
#
#model_ = 'nr_medellin_strategy'
#
#predictions_df = df_validation["id"].to_frame()
#predictions_df[model_] = preds
#predictions_df.to_csv(model_ +"_predictions.csv", index=False)




