
#basic imports
import json
from joblib import load


import numpy as np
import pandas as pd
import scipy

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression

#model funcs
#from prediction_scripts.base_predictions import get_predictions_l0_vanilla







##############################################################################################################################
##############################################################################################################################
##############################################################################################################################






def get_scores_feat_raw(data_type='live', score_type='mmc'):
	path = 'https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/scores_feat_raw/'
	scores_feat_raw = dict()
	# for score_type in ['era', 'mmc', 'tb200']:
	scores_path = path + score_type + '_scores_feat_raw_' + data_type + '.csv'
	# print(scores_path)
	scores_feat_raw = pd.read_csv(scores_path)
	return scores_feat_raw


def team_lr_fn(lr, fn):
	if lr > fn:
		return 1
	else:
		return -1


url_overall = 'https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/semanal/overall.csv'
def get_fn_strategy(url):
	fn_strategy_dict = pd.read_csv(url).to_dict()
	strategy_vegas = dict()
	pkl_name = list(fn_strategy_dict.keys())[0]
	for group, fn in fn_strategy_dict[pkl_name].items(): strategy_vegas[group] = [fn, 0]
	return strategy_vegas, pkl_name

import numerapi
def get_regime_dummy():
  napi = numerapi.NumerAPI()
  more_recent_round = napi.get_current_round()-1

  round_number = pd.DataFrame(napi.round_model_performances('tolsnr_15'))['roundNumber']
  mmc_series_ols = pd.DataFrame(napi.round_model_performances('tolsnr_15'), index=round_number)['mmc']
  mmc_series_fn = pd.DataFrame(napi.round_model_performances('xfnmn_15'), index=round_number)['mmc']
  #print(mmc_series_ols)
  if mmc_series_ols[more_recent_round] >=mmc_series_fn[more_recent_round]:
    return 1

  else:
    return -1


def get_dominance_eras(data_type, dom_sign, dom_source='git', era=None):
	# Get dominance era
	# era = df.era.unique()[-1]
	#dom_source='dummy'

	if data_type == 'live':
		if dom_source == 'git':
			dom, _ = get_fn_strategy(url_overall)

			if dom ==0:
				dom = get_regime_dummy()


		elif dom_source == 'dummy':
			dom = get_regime_dummy()

		dominance_eras = pd.Series([dom['dominance'][0]], index=[era])

	else:
		score_feat_raw = get_scores_feat_raw(data_type=data_type)
		dominance_eras = score_feat_raw[['lr', 'ex_FN100']].apply(
			lambda x: team_lr_fn(x['lr'], x['ex_FN100']), axis=1)

		# get previous dom
		if dom_source == 'dummy':
			dominance_eras = dominance_eras.shift(1)
			dominance_eras.iloc[0] = dominance_eras.iloc[1]

	# reverse dominance or not
	dominance_eras = dominance_eras * dom_sign
	return dominance_eras



def get_dominance_model(era, dom_sign, data_type, dom_source, model_preds_type):

	dominance_eras = get_dominance_eras(data_type, dom_sign, dom_source, era)
	dom_era = dominance_eras[era]
	#print(model_preds_type)

	if model_preds_type=='dom_preds_xgb':
		if dom_era==1:
			model_preds = 'xgb_dom_lr'
		else:
			model_preds = 'xgb_dom_fn'

	elif model_preds_type=='dom_preds_lr':
		if dom_era==1:
			model_preds = 'lr_dom_lr'
		else:
			model_preds = 'xgb_1_120'

	elif model_preds_type=='reg_preds':
		if dom_era==1:
			model_preds = 'lr_1_120'
		else:
			model_preds = 'xgb_1_120'

	else:
		model_preds = model_preds_type

	#print(model_preds)
	return [model_preds], dom_era



def fexp_feats_to_fn_era(df, dom_sign, data_type, dom_source):
	# get data
	features = [x for x in df.columns if x.startswith('feature_')]

	era = df.era.unique()[-1]

	score_feat_raw = get_scores_feat_raw(data_type='train')
	corr_scores = score_feat_raw.corr()

	feat_team_lr_fn = corr_scores[['lr', 'ex_FN100']].apply(
		lambda x: team_lr_fn(x['lr'], x['ex_FN100']), axis=1)

	# get dominance
	dominance_eras = get_dominance_eras(data_type, dom_sign, dom_source, era)

	# get features
	fexp_col = 'lr'
	# lr_exp_feats = fast_score_by_date(df, features, fexp_col, tb=200, era_col='era')
	lr_exp_feats = df.groupby("era").apply(lambda d: d[features].corrwith(df[fexp_col]))

	feat_team_eras = lr_exp_feats.apply(np.sign) * feat_team_lr_fn[features]
	feat_team_eras_values = lr_exp_feats * feat_team_lr_fn[features]

	# feat netr by era
	feats_neutr_era_df = feat_team_eras.apply(lambda x: x != dominance_eras[era], axis=0).T
	feats_to_neutr_era = list(feats_neutr_era_df[feats_neutr_era_df[era] == True].index)

	return lr_exp_feats, feat_team_eras, feat_team_eras_values, feats_to_neutr_era, dominance_eras


def feats_to_short(exposures_era, feats_fn_era, k):
	sort_feat_exp = abs(exposures_era[feats_fn_era])
	least_k = int(len(feats_fn_era) * (k))

	if least_k == 0: least_k = 1

	feats_short = sort_feat_exp.sort_values()[-least_k:].index
	feats_fn = sort_feat_exp.sort_values()[:-least_k].index

	return feats_fn, feats_short


def strategy_single(params, p):
	factor_by_dict = dict()
	era = params[0]
	feats_to_neutr_era = params[4]
	dominance_eras = params[5]

	factor_by_dict[p] = feats_to_neutr_era
	print('era{}. {}D, {}N, {}S'.format(
		era, dominance_eras[era], len(feats_to_neutr_era), 0))

	return factor_by_dict


def strategy_single_OH(params, p, map):
	factor_by_dict = dict()
	feats_OH = dict()
	era = params[0]
	feats_to_neutr_era = params[4]
	dominance_eras = params[5]
	# print('map: ', map)

	for key, feat_value in map.items():
		feats_OH = [i + '_' + str(key) for i in feats_to_neutr_era]
		factor_by_dict[p[key]] = {feat_value: feats_OH}

	print('era{}. {}D, {}N, {}S'.format(
		era, dominance_eras[era], len(feats_to_neutr_era), 0))

	return factor_by_dict

def strategy_single_short(params, k, p):
	factor_by_dict = dict()
	era = params[0]
	feat_team_eras_values = params[3]
	feats_to_neutr_era = params[4]
	dominance_eras = params[5]

	feats_fn, feats_short = feats_to_short(feat_team_eras_values.T[era], feats_to_neutr_era, k)

	factor_by_dict[p[0]] = feats_fn
	factor_by_dict[p[1]] = feats_short

	print('era{}. {}D, {}N, {}S'.format(
		era, dominance_eras[era], len(feats_fn), len(feats_short)))

	return factor_by_dict


def strategy_single_short_swap(params, p):
	factor_by_dict = dict()
	era = params[0]
	lr_exp_feats = params[1]
	feat_team_eras_values = params[3]
	feats_to_neutr_era = params[4]
	dominance_eras = params[5]

	fexp_sign = lr_exp_feats.apply(np.sign).T
	fexp_neg = fexp_sign[fexp_sign[era] == -1].index
	feats_to_short = fexp_neg.isin(feats_to_neutr_era)
	short_feats_era = list(fexp_neg[feats_to_short])
	neutr_feats_era = list(set(feats_to_neutr_era) - set(short_feats_era))

	factor_by_dict[p[0]] = neutr_feats_era
	factor_by_dict[p[1]] = short_feats_era

	print('era{}. {}D, {}N, {}S'.format(
		era, dominance_eras[era], len(neutr_feats_era), len(short_feats_era)))

	return factor_by_dict


def strategy_single_short_swap_fexp(params, k, p):
	factor_by_dict = dict()
	era = params[0]
	lr_exp_feats = params[1]
	feat_team_eras_values = params[3]
	feats_to_neutr_era = params[4]
	dominance_eras = params[5]

	fexp_sign = lr_exp_feats.apply(np.sign).T
	fexp_neg = fexp_sign[fexp_sign[era] == -1].index
	feats_to_short_selected = fexp_neg.isin(feats_to_neutr_era)
	short_feats_era = list(fexp_neg[feats_to_short_selected])

	_, short_feats_era = feats_to_short(feat_team_eras_values.T[era], short_feats_era, k)
	neutr_feats_era = list(set(feats_to_neutr_era) - set(short_feats_era))

	factor_by_dict[p[0]] = neutr_feats_era
	factor_by_dict[p[1]] = short_feats_era

	print('era{}. {}D, {}N, {}S'.format(
		era, dominance_eras[era], len(neutr_feats_era), len(short_feats_era)))

	return factor_by_dict


############################################################
############################################################
############################################################





def get_features_fs_lr(metric, lev, short):
	url = 'https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/feature_importance/fn_dom/'
	importances_df = pd.read_csv(url + 'features_neutralize_lr_eras.csv')

	# LR - lev
	criteria = importances_df[metric].quantile(lev[0])
	lev_feats = list(importances_df[importances_df[metric] < criteria].index)
	lev_feats = [c for c in lev_feats if c.startswith('feature_')]

	# LR - short
	criteria = importances_df[metric].quantile(1 - short[0])
	short_feats = list(importances_df[importances_df[metric] > criteria].index)
	short_feats = [c for c in short_feats if c.startswith('feature_')]

	by = {lev[1]: lev_feats, short[1]: short_feats}
	return by



def get_features_fs_fn(metric, lev, short):
	url = 'https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/feature_importance/fn_dom/'
	importances_df = pd.read_csv(url + 'features_neutralize_fn_eras.csv')

	# FN - lev
	criteria = importances_df[metric].quantile(1 - lev[0])
	lev_feats = list(importances_df[importances_df[metric] > criteria].index)
	lev_feats = [c for c in lev_feats if c.startswith('feature_')]

	# FN- short
	criteria = importances_df[metric].quantile(short[0])
	short_feats = list(importances_df[importances_df[metric] < criteria].index)
	short_feats = [c for c in short_feats if c.startswith('feature_')]

	by = {lev[1]: lev_feats, short[1]: short_feats}
	return by




def get_features_groups(groups):
	url = 'https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/feature_importance/fn_dom/'
	df = pd.read_csv(url + 'features_neutralize_lr_eras.csv').T

	dict_by = dict()
	for group_by, p in groups.items():
		dict_by[p[0]] = [c for c in df if c.startswith('feature_' + group_by)]
	return dict_by



def get_features_metric(func, metric, k, p):
	dict_by = dict()
	dict_by[p] = func(metric, k)

	return dict_by


############################################################
############################################################
############################################################


def strategy_regime(dom, lr, fn):
	if dom == 1:
		factor_by_dict = lr['dom_func'](**lr['dom_params'])

	else:
		factor_by_dict = fn['dom_func'](**fn['dom_params'])

	return factor_by_dict



def feats_to_neutralize_regime(df, dom_sign, data_type, dom_source, **strategy):
	era = df.era.unique()[-1]
	dominance_eras = get_dominance_eras(data_type, dom_sign, dom_source, era)
	dom = dominance_eras[era]
	#print('era: {},  dom: {}'.format(era, dom))

	# strategy_regime
	factor_by_strategy_dict = strategy['strategy_func'](dom, **strategy['strategy_params'])

	return factor_by_strategy_dict


def strategy_dominance(params, lr, fn):
	era = params[0]
	dominance_eras = params[5]

	if dominance_eras[era] == 1:
		factor_by_dict = lr['dom_func'](params, **lr['dom_params'])

	else:
		factor_by_dict = fn['dom_func'](params, **fn['dom_params'])

	return factor_by_dict


def feats_to_neutralize_dynamic(df, dom_sign, data_type,dom_source, **strategy):
	# strategies
	lr_exp_feats, feat_team_eras, feat_team_eras_values, feats_to_neutr_era, dominance_eras = fexp_feats_to_fn_era(df,
																												   dom_sign,
																												   data_type,
																												   dom_source)
	era = df.era.unique()[-1]
	factor_by_strategy_dict = dict()

	factor_by_strategy_dict = strategy['strategy_func']([era,
														 lr_exp_feats,
														 feat_team_eras,
														 feat_team_eras_values,
														 feats_to_neutr_era,
														 dominance_eras],
														**strategy['strategy_params'])

	return factor_by_strategy_dict







############################################################
############################################################
############################################################

def fexp_all_feats_tb200_live(df, columns, target, tb=200, era_col="era"):

    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        era_pred = np.float64(df_era[columns].values.T)
        era_target = np.float64(df_era[target].values.T)

        if tb is None:
            ccs = np.corrcoef(era_target, era_pred)[0, 1:]
        else:
            tbidx = np.argsort(era_pred, axis=1)
            tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
            ccs = [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)]
            ccs = np.array(ccs)

        computed.append(ccs)

    return pd.DataFrame(np.array(computed), columns=columns, index=df[era_col].unique())





def fexp_all_feats_live(df, features, column):

	fexp_preds_all_feats = df.groupby('era').apply(lambda d: d[features].corrwith(df[column]))
	return fexp_preds_all_feats


def fimp_all_feats_train(features, model):

	if model=='xgb':
		xgb_model = load('trained_models/l0_vanilla/xgb_1_120-cv.pkl')
		fimp = xgb_model.model.feature_importances_

	elif model=='lr':
		lr_model = load('trained_models/l0_vanilla/lr_1_120-cv.pkl')
		fimp = lr_model.model.coef_

	fimp_all_feats = pd.Series(abs(fimp), index=features)
	return fimp_all_feats


def feature_metrics_train_from_json(feature, root_path='json/one_feat/'):

	json_file = json.load(open(root_path+feature+'.json'))
	feat_scores = pd.read_json(json_file['metrics'])["Valor"]  # orient='index'

	#choose metrics
	metrics = ['Validation_SD', 'Max_Drawdown', 'VaR_10', 'AR(1)_sign', 'tb200_std']
	return pd.DataFrame(feat_scores[metrics]).T



def fexp_train_from_json(model, feature, root_path='json/raw/'):

	feat_exps_df = pd.DataFrame([])
	json_file = json.load(open(root_path + model + '.json'))
	feat_exps = pd.read_json(json_file['feat_corrs'])[feature]  # orient='index'
	tb200_exp = pd.read_json(json_file['tb200_exp'])[feature]  # orient='index'

	#get fexp metrics
	feat_exps_df['fexp_median_'+model] = [feat_exps.median()]
	feat_exps_df['fexp_std_'+model] = [feat_exps.std()]

	#get tb200 exp metrics
	feat_exps_df['tb200_exp_median_'+model] = [tb200_exp.median()]
	feat_exps_df['tb200_exp_std_'+model] = [tb200_exp.std()]

	return feat_exps_df

def fexp_live_from_preds(model, feature, **fexp_all_feats):
	live_feat_params = pd.DataFrame()

	live_feat_params[model+'_feat_corr'] = fexp_all_feats[model+'_feat_corrs'][feature]
	live_feat_params[model+'_tb200_corr'] = fexp_all_feats[model+'_tb200_corrs'][feature]

	live_feat_params[model+'_diff_corr'] = \
		live_feat_params[model+'_tb200_corr'] - live_feat_params[model+'_feat_corr']

	live_feat_params[model+'_rank_feat_corr'] = \
		fexp_all_feats[model + '_feat_corrs'].T.rank(pct=True).T[feature]

	live_feat_params[model+'_rank_tb200_corr'] = \
		fexp_all_feats[model + '_tb200_corrs'].T.rank(pct=True).T[feature]

	#FIMP
	live_feat_params[model + '_fimp'] = fexp_all_feats[model + '_fimp'][feature]


	return live_feat_params


def pos_neg(x):
		if x <= 0:
			return 0
		else:
			return 1

def create_target(feature):
	path = "https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/predicoes_validacao/shanghai/"
	era_scores = pd.read_csv(path + 'shanghai_preds_corr/era_scores/era_scores_train_target.csv')
	target = era_scores[feature].apply(lambda x: pos_neg(x))
	return target



def concat_synthetic_data_feature(model, feature, **fexp_all_feats):


	#get fexp metrics train
	fexp_metrics_train = fexp_train_from_json(model, feature)

	#get live metrics
	fexp_metrics_live = fexp_live_from_preds(model, feature, **fexp_all_feats)



	df_metrics_feature = pd.concat([
									fexp_metrics_train.reset_index(drop=True, inplace=False),
									fexp_metrics_live.reset_index(drop=True, inplace=False)
									], ignore_index= False, axis=1)


	return df_metrics_feature


def concat_synthetic_data_model(feature, **fexp_params):

	#get feature risk metrics
	feature_metrics_train = feature_metrics_train_from_json(feature)

	df_metrics_feature_lr = concat_synthetic_data_feature('lr', feature, **fexp_params)
	df_metrics_feature_xgb = concat_synthetic_data_feature('xgb', feature, **fexp_params)

	df_metrics_feature = pd.concat([
									feature_metrics_train.reset_index(drop=True, inplace=False),
									df_metrics_feature_lr.reset_index(drop=True, inplace=False),
									df_metrics_feature_xgb.reset_index(drop=True, inplace=False)
									], ignore_index= False, axis=1)


	return df_metrics_feature



def save_feature_era_data(data, era, feature):
	# append data era + info
	pass


def build_predictions_l0_models(df_predict, model_name_pkl):
	#print('bbbb')
	return False






def concat_synthetic_data_era(ddf, live_era=None, save_data=False, predict=True, *hparams):
	df = ddf.copy()
	features = [x for x in df.columns if x.startswith('feature_')]
	selected_features = []

	# get live era with predictions
	#df['preds_xgb'] = get_predictions_l0_vanilla('xgb_1_120')
	#df['preds_lr'] = get_predictions_l0_vanilla('lr_1_120')

	# get live era
	if live_era is None: live_era = df.era.unique()[-1]

	# slice the data
	df_live_era = df[df.era == live_era]

	fexp_params = dict()
	fexp_params['lr_feat_corrs'] = fexp_all_feats_live(df_live_era, features, 'preds_lr')
	fexp_params['xgb_feat_corrs'] = fexp_all_feats_live(df_live_era, features, 'preds_xgb')
	fexp_params['lr_tb200_corrs'] = fexp_all_feats_tb200_live(df_live_era, features, 'preds_lr')
	fexp_params['xgb_tb200_corrs'] = fexp_all_feats_tb200_live(df_live_era, features, 'preds_xgb')
	fexp_params['lr_fimp'] = fimp_all_feats_train(features, 'lr')
	fexp_params['xgb_fimp'] = fimp_all_feats_train(features, 'xgb')



	for feature in features:
		df_metrics_feature_era = concat_synthetic_data_model(feature, **fexp_params)

		if save_data==True:

			df_metrics_feature_era['target'] = create_target(feature)[live_era]
			df_metrics_feature_era['era'] = live_era
			save_feature_era_data(df_metrics_feature_era, live_era, feature)

		if predict==True:
			feat_approve = build_predictions_l0_models(df_metrics_feature_era, feature)

			if feat_approve==True:
				selected_features.append(feature)

		print(df_metrics_feature_era.T)
		break



	print(hparams)
	return ['feature_charisma1']#selected_features




# funcao para chamar em producao
def fs_dynamic(ddf, *hparams):
	return concat_synthetic_data_era(ddf, None, True, True, *hparams)
