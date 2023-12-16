import numpy as np
import pandas as pd
import scipy

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

try:
    from src.models import fs_dynamic_synthetic
except:
    import fs_dynamic_synthetic


def _neutralize(df, columns, by, ml_model, proportion):  # ['preds'], features,
    scores = df[columns]  # preds
    exposures = df[by].values  # features

    if proportion[0] == 'non_linear':
        # print('non_linear')
        xgb_model = XGBRegressor(max_depth=5, learning_rate=0.01,
                                 n_estimators=2000, n_jobs=-1,
                                 colsample_bytree=0.1,
                                 # tree_method='gpu_hist', gpu_id=0
                                 )

        xgb_model.fit(exposures, scores.values.reshape(1, -1)[0])
        neutr_preds = pd.DataFrame(xgb_model.predict(exposures), index=df.index, columns=columns)
        proportion[0] = 1
        # print('predicted')

    else:
        ml_model[0].fit(exposures, scores.values.reshape(1, -1)[0])
        neutr_preds = pd.DataFrame(ml_model[0].predict(exposures), index=df.index, columns=columns)
    # exposures.dot(np.linalg.pinv(exposures).dot(scores))

    if proportion[1] != 0.0:
        ml_model[1].fit(exposures, scores.values.reshape(1, -1)[0])
        neutr_preds2 = pd.DataFrame(ml_model[1].predict(exposures), index=df.index, columns=columns)
        # print(neutr_preds2)

    else:
        neutr_preds2 = 0  # np.zeros(len(scores))

    scores = scores - ((proportion[0] * neutr_preds) + ((proportion[1]) * neutr_preds2))

    # scores = scores - proportion * neutr_preds
    return scores / scores.std()


def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return scipy.stats.norm.ppf(X)


def normalize_and_neutralize(df, columns, by, ml_model, proportion):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, ml_model, proportion)
    return df[columns]


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

# todas as colunas de uma vez
def preds_neutralized_old(ddf, columns, by, ml_model, proportion):
    df = ddf.copy()
    preds_neutr = df.groupby("era").apply(
        lambda x: normalize_and_neutralize(x, columns, by, ml_model, proportion))

    preds_neutr = MinMaxScaler().fit_transform(preds_neutr).reshape(1, -1)[0]

    return preds_neutr


# por grupo mas na mesma proporcao
def preds_neutralized(ddf, columns, by, ml_model, proportion):
    df = ddf.copy()
    preds_neutr = dict()
    for group_by in by:
        feat_by = [c for c in df if c.startswith('feature_' + group_by)]

        df[columns] = df.groupby("era").apply(
            lambda x: normalize_and_neutralize(x, columns, feat_by, ml_model, proportion))

        preds_neutr_after = MinMaxScaler().fit_transform(df[columns]).reshape(1, -1)[0]

    return preds_neutr_after


def preds_neutralized_groups(ddf, columns, by, ml_model, _):
    df = ddf.copy()

    for group_by, p in by.items():
        feat_by = [c for c in df if c.startswith('feature_' + group_by)]

        df[columns] = df.groupby("era").apply(
            lambda x: normalize_and_neutralize(x, columns, feat_by, ml_model, [p[0], p[1]]))

        preds_neutr_after = MinMaxScaler().fit_transform(df[columns]).reshape(1, -1)[0]

    return preds_neutr_after


def feature_exposure_old(df, pred):
    # df = df[df.data_type == 'validation']
    pred = pd.Series(pred, index=df.index)

    feature_columns = [x for x in df.columns if x.startswith('feature_')]
    correlations = []
    for col in feature_columns:
        correlations.append(np.corrcoef(pred.rank(pct=True, method="first"), df[col])[0, 1])
    corr_series = pd.Series(correlations, index=feature_columns)
    return np.std(correlations), max(np.abs(correlations)), corr_series


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


def ar1(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def ar1_sign(x):
    return ar1((x > np.mean(x)) * 1)


def autocorr_penalty(x, sign=False):
    n = len(x)
    if sign == True:
        p = np.abs(ar1_sign(x))
    else:
        p = np.abs(ar1(x))

    return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p ** i for i in range(1, n)]))


def smart_sharpe_abs_sign(x):  ##ADD minus to use FS_AR1_SIGN
    return (abs(np.mean(x)) / (np.std(x, ddof=1) * autocorr_penalty(x, sign=True)))  # * np.sqrt(12))

def max_drawdown(x):

    rolling_max = (x + 1).cumprod().rolling(window=100,min_periods=1).max()
    daily_value = (x + 1).cumprod()

    max_dd = -((rolling_max - daily_value) / rolling_max).max()

    #print(f"max drawdown: {max_dd}")
    return max_dd

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


def neutralize_topk_era(df, preds, k, ml_model, proportion):
    _, _, feat_exp = feature_exposure_old(df, preds)

    if proportion[0] == 'std':
        proportion = [proportion[1], 0]
        path = "https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/predicoes_validacao/shanghai/"
        era_scores = pd.read_csv(path + 'shanghai_preds_corr/era_scores/era_scores_train_target.csv')
        mode_series = era_scores.apply(lambda x: np.std(x))  # .sort_values(ascending=False)
        fexp = feat_exp.abs() * mode_series

    else:
        fexp = feat_exp.abs()
    k_exposed = feat_exp[fexp > fexp.quantile(1 - k)].index

    preds_era = preds_neutralized_old(df, ['preds_fn'], k_exposed, ml_model, proportion)

    return preds_era


def neutralize_topk(ddf, preds, k, ml_model, proportion):
    df = ddf.copy()
    df['preds_fn'] = preds

    preds_neutr_topk_era = df.groupby("era", sort=False).apply(
        lambda x: neutralize_topk_era(x, x['preds_fn'], k, ml_model, proportion))

    return np.hstack(preds_neutr_topk_era)


# from typing import Callable


def fs_ar1_sign(metric, k, scores='era'):  # pega os maiores
    path = "https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/predicoes_validacao/shanghai/"
    era_scores = pd.read_csv(path + 'shanghai_preds_corr/era_scores/' + scores + '_scores_train_target.csv')

    mode_series = era_scores.apply(lambda x: metric(x)).sort_values(ascending=False)
    feats = mode_series[mode_series.abs() > mode_series.abs().quantile(1 - k)].index

    return feats


def fs_sharpe(metric, k, scores='era'):  # pega os menores
    path = "https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/predicoes_validacao/shanghai/"
    era_scores = pd.read_csv(path + 'shanghai_preds_corr/era_scores/' + scores + '_scores_train_target.csv')

    mode_series = era_scores.apply(lambda x: metric(x)).sort_values(ascending=False)
    feats = mode_series[mode_series.abs() < mode_series.abs().quantile(k)].index

    return feats


def fs_corr(_, k, __):  # serve apenas para correlacao

    tb200 = 'tb200'
    era = 'era'
    scores_dict = dict()
    path = "https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/predicoes_validacao/shanghai/"
    scores_dict[era] = pd.read_csv(path + 'shanghai_preds_corr/era_scores/' + era + '_scores_train_target.csv')
    scores_dict[tb200] = pd.read_csv(path + 'shanghai_preds_corr/era_scores/' + tb200 + '_scores_train_target.csv')

    models = list(scores_dict['era'].columns)
    flat_corr_matrix = pd.DataFrame(index=['era_tb'])  # , 'era_mmc', 'mmc_tb'])

    for model in models:
        temp_df = pd.DataFrame([])
        for scores_type in ['era', 'tb200']:  # , 'mmc_scores']:
            temp_df[scores_type] = scores_dict[scores_type][model]
        flat_matrix = temp_df.corr().values.flatten()
        flat_corr_matrix[model] = [flat_matrix[1]]  # +[flat_matrix[2]]+[flat_matrix[5]] #3x3 Matrix
        mode_series = flat_corr_matrix.T.era_tb.sort_values(ascending=False)
        feats = mode_series[mode_series.abs() < mode_series.abs().quantile(k)].index
    return feats


def fs_sfi(model_fs, qt=.3):  # , ranker=True):

    url = 'https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/feature_importance/'
    importances_df = pd.read_csv(url + model_fs + '.csv')
    # importances_df = importances_df.reindex(df[])

    if model_fs[:10] == 'linear/mdi': criteria = 1 / importances_df.shape[0]
    if model_fs[:10] == 'linear/mda': criteria = 0
    if model_fs[:10] == 'linear/sfi': criteria = importances_df['mean'].quantile(qt)

    if model_fs[:21] == 'cluster/clustered_mdi': criteria = 1 / importances_df.shape[0]
    if model_fs[:21] == 'cluster/clustered_mda': criteria = 0

    if model_fs[:25] == 'dss/cluster/all_feats/mda': criteria = 0

    if model_fs[:14] == 'dss/regime/sfi': criteria = 0
    if model_fs[:14] == 'dss/regime/mda': criteria = 0

    features = list(importances_df.index)
    importances_df = importances_df[importances_df['mean'] > criteria]
    features_selected = list(importances_df.index)

    # if ranker ==True: features_selected = ['era']+features_selected
    features_neutralize = list(set(features) - set(features_selected))

    return features_neutralize


def fs_ebm_reg(model_fs, qt=.3):  # , ranker=True):

    url = 'https://raw.githubusercontent.com/nbarbosa-git/compute_info/main/public_semanal/feature_importance/'
    # url = '../../reports/feature_importance/'

    importances_df = pd.read_csv(url + model_fs + '.csv')
    # importances_df = importances_df.reindex(df[])

    if model_fs[:] == 'shap/vanilla': criteria = -1
    if model_fs[:] == 'shap/full_FN': criteria = 1
    if model_fs[:8] == 'shap/ebm': criteria = importances_df['mean'].quantile(qt)  # 1/importances_df.shape[0]
    if model_fs[:11] == 'shap/morris': criteria = importances_df['mean'].quantile(qt)
    if model_fs[:9] == 'shap/shap': criteria = importances_df['mean'].quantile(qt)
    if model_fs[:21] == 'cluster/clustered_mdi': criteria = 1 / importances_df.shape[0]
    if model_fs[:21] == 'cluster/clustered_mda': criteria = 0

    # print('criteria: ', criteria)
    features = list(importances_df.index)
    importances_df = importances_df[importances_df['mean'] > criteria]
    features_selected = list(importances_df.index)

    # if ranker ==True: features_selected = ['era']+features_selected
    features_neutralize = list(set(features) - set(features_selected))
    # print('nao neutralizarei: ', len(features_selected))
    # print('neutralizarei: ', features_selected)

    return features_neutralize


def preds_neutralized_fs(ddf, columns, func, param_func, ml_model, p):
    df = ddf.copy()

    if param_func[0] == 'dynamic':
        feats = func(df, *param_func)

    else:
        feats = func(*param_func)

    by = {p[0]: feats}

    # print(str(param_func[0]))
    # print(len(feats))
    # print(feats)

    for p, feat_by in by.items():
        df[columns] = df.groupby("era").apply(lambda x: normalize_and_neutralize(x, columns, feat_by, ml_model, [p, 0]))
        preds_neutr_after = MinMaxScaler().fit_transform(df[columns]).reshape(1, -1)[0]

    return preds_neutr_after


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


from sklearn.preprocessing import OneHotEncoder


def nn_OH(df, columns, feat_value, ml_model, proportion):
    features = [x for x in df.columns if x.startswith('feature_')]

    enc = OneHotEncoder(sparse=False)
    enc_array = enc.fit_transform(df[features])
    enc_features = enc.get_feature_names(features)

    df_enc = pd.DataFrame(enc_array, columns=enc_features, index=df.index)
    df_enc['era'] = df.era.values
    df_enc[columns[0]] = df[columns[0]].values

    feat_by = enc_features[feat_value::5]

    df[columns] = normalize_and_neutralize(df_enc, columns, feat_by, ml_model, proportion)
    return df[columns]


def preds_neutralized_one_hot(ddf, columns, by, ml_model, p):
    df = ddf.copy()
    df_OH = df.copy()

    for key, feat_value in by.items():
        df_OH[columns] = df_OH.groupby("era", sort=False).apply(
            lambda x: nn_OH(x, columns, feat_value, ml_model, p[key]))
        preds_neutr_after = MinMaxScaler().fit_transform(df_OH[columns]).reshape(1, -1)[0]

    return preds_neutr_after


#############


def nn_OH_FS(df, columns, feat_value, feats, ml_model, proportion):
    features = [x for x in df.columns if x.startswith('feature_')]

    enc = OneHotEncoder(sparse=False)
    enc_array = enc.fit_transform(df[features])
    enc_features = enc.get_feature_names(features)

    df_enc = pd.DataFrame(enc_array, columns=enc_features, index=df.index)
    df_enc['era'] = df.era.values
    df_enc[columns[0]] = df.preds.values

    feat_by = enc_features[feat_value::5]
    selected_feats_OH = [x for x in feat_by if x in feats]
    # print(type(selected_feats_OH))
    # print(type(feat_by))

    df[columns] = normalize_and_neutralize(df_enc, columns, selected_feats_OH, ml_model, proportion)
    return df[columns]


def preds_neutralized_one_hot_FS(ddf, columns, func, param_func, oh_map, ml_model, p):
    df = ddf.copy()
    df_OH = df.copy()

    feats = func(*param_func)
    by = oh_map
    # print(feats)

    for key, feat_value in by.items():
        feats_OH = [i + '_' + str(key) for i in feats]
        # print(feats_OH)
        df_OH[columns] = df_OH.groupby("era", sort=False).apply(
            lambda x: nn_OH_FS(x, columns, feat_value, feats_OH, ml_model, p[key]))
        preds_neutr_after = MinMaxScaler().fit_transform(df_OH[columns]).reshape(1, -1)[0]

    return preds_neutr_after


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


def preds_neutralized_fs_dyn(ddf, columns, func, param_func, ml_model, factor):
    df = ddf.copy()
    by = func(df, *param_func[:3], **factor)

    era = df.era.unique()[-1]
    columns, dom = fs_dynamic_synthetic.get_dominance_model(era, *param_func)
    #print(columns)
    #print(df['preds'])
    df['preds'] = df[columns[0]]


    dom_dict = {1: 'lr', -1: 'fn'}

    if factor['strategy_params'][dom_dict[dom]]['dom_func'] == fs_dynamic_synthetic.strategy_single_OH:
        df_OH = df.copy()

        for p, feat_by in by.items():
            feat_value = list(feat_by.keys())[0]
            feats_OH = feat_by[feat_value]
            # print('feat value: ', feat_by.keys())

            df['preds'] = df.groupby("era").apply(
                lambda x: nn_OH_FS(x, columns, feat_value, feats_OH, ml_model, [p, 0]))
            # preds_neutr_after = MinMaxScaler().fit_transform(df[columns]).reshape(1, -1)[0]

    else:
        for p, feat_by in by.items():
            #print("len:{}, ft:{}".format(len(feat_by),feat_by))
            df['preds'] = df.groupby("era").apply(
                lambda x: normalize_and_neutralize(x, ['preds'], feat_by, ml_model, [p, 0]))
            # preds_neutr_after = MinMaxScaler().fit_transform(df[columns]).reshape(1, -1)[0]

    return df['preds']  # pd.DataFrame(preds_neutr_after, index=df.index)




def preds_neutralized_fs_dyn_era(ddf, columns, func, param_func, ml_model, factor):
    df = ddf.copy()

    if df.era.unique()[-1] == 9999:  # 9999 961
        param_func[1] = 'live'

    preds = df.groupby("era").apply(
        lambda x: preds_neutralized_fs_dyn(x, columns, func, param_func, ml_model, factor))

    pp = preds.values.reshape(-1, 1)
    # print('preds: ', pp)
    preds_neutr_after = MinMaxScaler().fit_transform(pp).reshape(1, -1)[0]
    # print(preds_neutr_after)
    return preds_neutr_after


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


fn_strategy_dict = {

    'ex_preds': {'strategy': 'after',
                 'func': preds_neutralized,
                 'columns': ['preds'],
                 'by': [''],
                 'model': [LinearRegression(), None],
                 'factor': [0.0, 0.0]
                 },

    'ex_FN100': {'strategy': 'after',
                 'func': preds_neutralized,
                 'columns': ['preds'],
                 'by': [''],
                 'model': [LinearRegression(), None],
                 'factor': [1.0, 0.0]
                 },

    'nr_home': {'strategy': None,
                'func': None,
                'columns': ['preds'],
                'by': [''],
                'model': [None, None],
                'factor': [0, 0]
                },

    'nr_vegas': {'strategy': 'after',
                 'func': preds_neutralized,
                 'columns': ['preds'],
                 'by': [''],
                 'model': [LinearRegression(), None],
                 'factor': [0.0, 0.0] #OVERALL
                },

    # R1
    'nr_buenos_aires': {'strategy': 'after',
                        'func': preds_neutralized,
                        'columns': ['preds'],
                        'by': [''],
                        'model': [SGDRegressor(tol=0.001), None],
                        'factor': [0.4, 0.0]
                        },

    'nr_rio_de_janeiro': {'strategy': 'after',
                          'func': preds_neutralized,
                          'columns': ['preds'],
                          'by': [''],
                          'model': [LinearRegression(), None],
                          'factor': [1.0, 0.0]
                          },

    'nr_sao_paulo': {'strategy': 'after',
                     'func': preds_neutralized,
                     'columns': ['preds'],
                     'by': [''],
                     'model': [SGDRegressor(tol=0.001), None],
                     'factor': [0.9, 0.0]
                     },

    'nr_medellin': {'strategy': 'after',
                    'func': preds_neutralized,
                    'columns': ['preds'],
                    'by': [''],
                    'model': [LinearRegression(), None],
                    'factor': [0.0, 0.0]  # OVERALLL
                    },

    'nr_guadalajara': {'strategy': 'after',
                       'func': preds_neutralized,
                       'columns': ['preds'],
                       'by': ['intelligence', 'constitution'],
                       'model': [LinearRegression(), None],
                       'factor': [1.0, 0.0]
                       },


    'nr_san_francisco': {'strategy': 'after',
                         'func': preds_neutralized_groups,
                         'columns': ['preds'],

                         'by': {'constitution': [1.5, 0.0], 'strength': [1.0, 0.0],
                                'dexterity': [-1.0, 0.0], 'charisma': [-1.0, 0.0],
                                'wisdom': [1.5, 0.0], 'intelligence': [1.0, 0.0]},

                         'model': [LinearRegression(), None],
                         'factor': []
                         },





    'nr_shanghai': {'strategy': 'double_fn',
                    'func': preds_neutralized_groups,
                    'columns': ['preds'],

                    'by': {'constitution': [1.5, 0.0], 'strength': [1.0, 0.0],
                           'dexterity': [0.0, 0.0], 'charisma': [0.0, 0.0],
                           'wisdom': [1.5, 0], 'intelligence': [1.0, 0.0]},

                    'model': [LinearRegression(), None],
                    'factor': [],

                    'double': {'func2': neutralize_topk,
                               'params2': 0.10,
                               'factor2': [0.25, 0.0],
                               'columns_fn': ['preds_fn'],
                               'model2': [LinearRegression(), None]},
                    },

    'nr_bangalore': {'strategy': 'after',
                     'func': preds_neutralized_fs,
                     'columns': ['preds'],
                     'by': [fs_ar1_sign, [ar1_sign, .1]],
                     'model': [LinearRegression(), None],
                     'factor': [1]
                     },

    'nr_hanoi': {'strategy': 'after',
                 'func': preds_neutralized_one_hot,
                 'columns': ['preds'],
                 'by': {0.0: 0},#1.0: 4},
                 'model': [SGDRegressor(tol=0.001), None],
                 'factor': {0.0: [.9, 0]}# 1.0: [.9, 0]}
                 },

    'nr_johannesburg': {'strategy': 'after',
                        'func': preds_neutralized_fs,
                        'columns': ['preds'],
                        'by': [fs_sfi, ['linear/sfi_vanilla']], #mda_linear_reg_cv sfi_vanilla
                        'model': [LinearRegression(), None],
                        'factor': [1]
                        },

    'nr_lagos': {'strategy': 'after',
                        'func': preds_neutralized_fs,
                        'columns': ['preds'],
                        'by': [fs_sfi, ['dss/cluster/all_feats/mda_xgb_denoise']],  # mda_linear_reg_cv sfi_vanilla
                        'model': [LinearRegression(), None],
                        'factor': [1]
                        },

    'nr_hong_kong': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git', 'preds']], ###
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_groups,
                              'dom_params': {'groups':{
                                  'constitution': [1.0, 0], 'strength': [1.0, 0],
                                  'dexterity': [0.0, 0],    'charisma': [0.0, 0],
                                  'wisdom': [0.0, 0],       'intelligence': [0.0, 0]}}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_groups,
                              'dom_params': {'groups':{
                                  'constitution': [0.0, 0], 'strength': [0.0, 0],
                                  'dexterity': [1.0, 0],    'charisma': [0.0, 0],
                                  'wisdom': [1.0, 0],       'intelligence': [-0.5, 0]}}},
                   }}},

    'nr_singapore': {'strategy': 'double_fn',
                     'func': preds_neutralized_groups,
                     'columns': ['preds'],

                     'by': {'constitution': [0.0, 0.0], 'strength': [0.0, 0.0],
                            'dexterity': [0.0, 0.0], 'charisma': [0.0, 0.0],
                            'wisdom': [0.0, 0], 'intelligence': [0.0, 0.0]},

                     'model': [LinearRegression(), LinearRegression()],
                     'factor': [],

                     'double': {'func2': neutralize_topk,
                                'params2': 0.10,
                                'factor2': [0.0, 0.0],
                                'columns_fn': ['preds_fn'],
                                'model2': [LinearRegression(), None]},
                     },

    'nr_bombay': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git', 'preds']],
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_metric,
                              'dom_params': {'k': .4, 'p': 0.75,
                                             'func': fs_sharpe,  # menores
                                             'metric': smart_sharpe_abs_sign}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_metric,
                              'dom_params': {'k': .5, 'p': .75,
                                             'func': fs_ar1_sign,  # maiores
                                             'metric': max_drawdown}},
                   }}},

    'nr_istanbul': { #strategy_sfi_mda
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git', 'preds']],
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_metric,
                              'dom_params': {'k': None, 'p': 1.0,
                                             'func': fs_sfi,
                                             'metric': 'dss/regime/sfi_xgb_easy'}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_metric,
                              'dom_params': {'k': None, 'p': 1.0,
                                             'func': fs_sfi,
                                             'metric': 'dss/regime/mda_lr_hard'}},
                   }}},


    'nr_kigali': {'strategy': 'after',
                        'func': preds_neutralized_fs,
                        'columns': ['preds'],
                        'by': [fs_sfi, ['dss/cluster/all_feats/mda_xgb_information_variation']],
                        'model': [LinearRegression(), None],
                        'factor': [1]
                        },


    'nr_saigon': {
        'strategy': 'after',
        'func': preds_neutralized_one_hot_FS,
        'columns': ['preds'],
        'by': [fs_ebm_reg,
               ['cluster/clustered_mda_iv_ward_tb200'],
               {0.0: 0}],

        'model': [SGDRegressor(tol=0.001), None],
        'factor': {0.0: [.9, 0]}
    },

    'nr_moscow': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git', 'preds']],
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_fs_lr,
                              'dom_params': {'metric': 'payout',
                                  'lev': [.001, 0], 'short': [0.05, 1.00]}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_fs_fn,
                              'dom_params': {'metric': 'payout',
                                  'lev': [.001, 0], 'short': [0.75, 1.00]}},
                   }}},


    'nr_tallinn': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git',  'dom_preds_xgb']],
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_fs_lr,
                              'dom_params': {'metric': 'payout',
                                  'lev': [.001, 0], 'short': [0.05, 1.00]}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_fs_fn,
                              'dom_params': {'metric': 'payout',
                                  'lev': [.001, 0], 'short': [0.75, 1.00]}},
                   }}},

    'nr_hamburg': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git', 'dom_preds_xgb']],
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_fs_lr,
                              'dom_params': {'metric': 'payout',
                                             'lev': [.05, -.5], 'short': [0.05, 1.25]}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_fs_fn,
                              'dom_params': {'metric': 'payout',
                                             'lev': [.05, -.5], 'short': [0.75, 1.25]}},
                   }}},


    'nr_stockholm': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git',  'reg_preds']], ###
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_groups,
                              'dom_params': {'groups':{'': [0, 0]}}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_groups,
                              'dom_params': {'groups':{'': [1, 0]}}},
                   }}},

    'nr_monaco': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git', 'dom_preds_lr']],
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_fs_lr,
                              'dom_params': {'metric': 'payout',
                                  'lev': [.05, -.5], 'short': [0.05, 1.25]}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_fs_fn,
                              'dom_params': {'metric': 'payout',
                                  'lev': [.05, -.5], 'short': [0.75, 1.25]}},
                   }}},



    'nr_barcelona': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git', 'dom_preds_xgb']],  ###
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_groups,
                              'dom_params': {'groups': {'': [0, 0]}}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_groups,
                              'dom_params': {'groups': {'': [0, 0]}}},
                   }}},



    # NO FN
    'nr_sydney': {'strategy': None, 'func': None, 'columns': ['preds'], 'by': [''], 'model': [None, None],
                  'factor': [0, 0]},


    'nr_san_salvador': {'strategy': None, 'func': None, 'columns': ['preds'], 'by': [''], 'model': [None, None],
                        'factor': [0, 0]},
    'nr_dubai': {'strategy': None, 'func': None, 'columns': ['preds'], 'by': [''], 'model': [None, None],
                 'factor': [0, 0]},
    'nr_havana': {'strategy': None, 'func': None, 'columns': ['preds'], 'by': [''], 'model': [None, None],
                  'factor': [0, 0]},
    'nr_zurich': {'strategy': None, 'func': None, 'columns': ['preds'], 'by': [''], 'model': [None, None],
                  'factor': [0, 0]},



    'strategy': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type|| Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_dynamic, [1, 'live', 'git', 'preds']],
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_dominance,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.strategy_single,
                              'dom_params': {'p': 1.0}},
                       'fn': {'dom_func': fs_dynamic_synthetic.strategy_single,
                              'dom_params': {'p': 1.0}},
                   }}},
    'minus_strategy': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type|| Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_dynamic, [-1, 'live', 'git', 'preds']],
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_dominance,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.strategy_single,
                              'dom_params': {'p': 1.0}},
                       'fn': {'dom_func': fs_dynamic_synthetic.strategy_single,
                              'dom_params': {'p': 1.0}},
                   }}},

    'new_dinamic': {
        'strategy': 'after',
        'func': preds_neutralized_fs_dyn_era,
        'columns': ['preds'],
        # Func || Dom Sign || Data Type || Dom Source || Dom preds
        'by': [fs_dynamic_synthetic.feats_to_neutralize_regime, [1, 'legacy', 'git', 'preds']],
        'model': [LinearRegression(), None],
        'factor': {'strategy_func': fs_dynamic_synthetic.strategy_regime,
                   'strategy_params': {
                       'lr': {'dom_func': fs_dynamic_synthetic.get_features_fs_lr,
                              'dom_params': {'metric': 'payout',
                                             'lev': [.05, -.5], 'short': [0.05, 1.25]}},

                       'fn': {'dom_func': fs_dynamic_synthetic.get_features_fs_fn,
                              'dom_params': {'metric': 'payout',
                                             'lev': [.05, -.5], 'short': [0.75, 1.25]}},
                   }}},

}
