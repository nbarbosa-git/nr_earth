import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def estimated_sharpe_ratio(returns):
    """
    Calculate the estimated sharpe ratio (risk_free=0).
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
    Returns
    -------
    float, pd.Series
    """
    return returns.mean() / returns.std(ddof=1)


def ann_estimated_sharpe_ratio(returns=None, periods=261, *, sr=None):
    """
    Calculate the annualized estimated sharpe ratio (risk_free=0).
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
    periods: int
        How many items in `returns` complete a Year.
        If returns are daily: 261, weekly: 52, monthly: 12, ...
    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio to be annualized, it's frequency must be coherent with `periods`
    Returns
    -------
    float, pd.Series
    """
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    sr = sr * np.sqrt(periods)
    return sr


def estimated_sharpe_ratio_stdev(returns=None, *, n=None, skew=None, kurtosis=None, sr=None):
    """
    Calculate the standard deviation of the sharpe ratio estimation.
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass the other 4 parameters.
    n: int
        Number of returns samples used for calculating `skew`, `kurtosis` and `sr`.
    skew: float, np.array, pd.Series, pd.DataFrame
        The third moment expressed in the same frequency as the other parameters.
        `skew`=0 for normal returns.
    kurtosis: float, np.array, pd.Series, pd.DataFrame
        The fourth moment expressed in the same frequency as the other parameters.
        `kurtosis`=3 for normal returns.
    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.
    Returns
    -------
    float, pd.Series
    Notes
    -----
    This formula generalizes for both normal and non-normal returns.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """
    if type(returns) != pd.DataFrame:
        _returns = pd.DataFrame(returns)
    else:
        _returns = returns.copy()

    if n is None:
        n = len(_returns)
    if skew is None:
        skew = pd.Series(scipy_stats.skew(_returns), index=_returns.columns)
    if kurtosis is None:
        kurtosis = pd.Series(scipy_stats.kurtosis(_returns, fisher=False), index=_returns.columns)
    if sr is None:
        sr = estimated_sharpe_ratio(_returns)

    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))

    if type(returns) == pd.DataFrame:
        sr_std = pd.Series(sr_std, index=returns.columns)
    elif type(sr_std) not in (float, np.float64, pd.DataFrame):
        sr_std = sr_std.values[0]

    return sr_std



def probabilistic_sharpe_ratio_(x=None, sr_benchmark=0.0):
    import scipy
    import math
    n = len(x)
    sr = np.mean(x) / np.std(x, ddof=1)
    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew(x) * sr) + (((kurtosis(x) - 3) / 4) * sr ** 2)) / (n - 1))
    psr = scipy.stats.norm.cdf((sr - sr_benchmark) / sr_std)
    if math.isnan(psr): psr=1.0000
    return psr

def probabilistic_sharpe_ratio(returns=None, sr_benchmark=0.0, *, sr=None, sr_std=None):
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass a `sr` and `sr_std`.
    sr_benchmark: float
        Benchmark sharpe ratio expressed in the same frequency as the other parameters.
        By default set to zero (comparing against no investment skill).
    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.
    sr_std: float, np.array, pd.Series, pd.DataFrame
        Standard deviation fo the Estimated sharpe ratio,
        expressed in the same frequency as the other parameters.
    Returns
    -------
    float, pd.Series
    Notes
    -----
    PSR(SR*) = probability that SR^ > SR*
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR* = `sr_benchmark`
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    psr = scipy_stats.norm.cdf((sr - sr_benchmark) / sr_std)

    if type(returns) == pd.DataFrame:
        psr = pd.Series(psr, index=returns.columns)
    elif type(psr) not in (float, np.float64):
        psr = psr[0]

    return psr


def min_track_record_length(returns=None, sr_benchmark=0.0, prob=0.95, *, n=None, sr=None, sr_std=None):
    """
    Calculate the MIn Track Record Length (minTRL).
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass a `sr` and `sr_std`.
    sr_benchmark: float
        Benchmark sharpe ratio expressed in the same frequency as the other parameters.
        By default set to zero (comparing against no investment skill).
    prob: float
        Confidence level used for calculating the minTRL.
        Between 0 and 1, by default=0.95
    n: int
        Number of returns samples used for calculating `sr` and `sr_std`.
    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.
    sr_std: float, np.array, pd.Series, pd.DataFrame
        Standard deviation fo the Estimated sharpe ratio,
        expressed in the same frequency as the other parameters.
    Returns
    -------
    float, pd.Series
    Notes
    -----
    minTRL = minimum of returns/samples needed (with same SR and SR_STD) to accomplish a PSR(SR*) > `prob`
    PSR(SR*) = probability that SR^ > SR*
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR* = `sr_benchmark`
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """
    if n is None:
        n = len(returns)
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    min_trl = 1 + (sr_std ** 2 * (n - 1)) * (scipy_stats.norm.ppf(prob) / (sr - sr_benchmark)) ** 2

    if type(returns) == pd.DataFrame:
        min_trl = pd.Series(min_trl, index=returns.columns)
    elif type(min_trl) not in (float, np.float64):
        min_trl = min_trl[0]

    return min_trl


def num_independent_trials(trials_returns=None, *, m=None, p=None):
    """
    Calculate the number of independent trials.
    
    Parameters
    ----------
    trials_returns: pd.DataFrame
        All trials returns, not only the independent trials.
        
    m: int
        Number of total trials.
        
    p: float
        Average correlation between all the trials.
    Returns
    -------
    int
    """
    if m is None:
        m = trials_returns.shape[1]
        
    if p is None:
        corr_matrix = trials_returns.corr()
        p = corr_matrix.values[np.triu_indices_from(corr_matrix.values,1)].mean()
        
    n = p + (1 - p) * m
    
    #try:
    n = int(n)+1  # round up
    #except:
    #n=1
    
    return n


def expected_maximum_sr(trials_returns=None, expected_mean_sr=0, *, independent_trials=None, trials_sr_std=None):
    """
    Compute the expected maximum Sharpe ratio (Analytically)
    
    Parameters
    ----------
    trials_returns: pd.DataFrame
        All trials returns, not only the independent trials.
        
    expected_mean_sr: float
        Expected mean SR, usually 0. We assume that random startegies will have a mean SR of 0,
        expressed in the same frequency as the other parameters.
        
    independent_trials: int
        Number of independent trials, must be between 1 and `trials_returns.shape[1]`
        
    trials_sr_std: float
        Standard deviation fo the Estimated sharpe ratios of all trials,
        expressed in the same frequency as the other parameters.
    Returns
    -------
    float
    """
    emc = 0.5772156649 # Euler-Mascheroni constant
    
    if independent_trials is None:
        independent_trials = num_independent_trials(trials_returns)
    
    if trials_sr_std is None:
        srs = estimated_sharpe_ratio(trials_returns)
        trials_sr_std = srs.std()
    
    maxZ = (1 - emc) * scipy_stats.norm.ppf(1 - 1./independent_trials) + emc * scipy_stats.norm.ppf(1 - 1./(independent_trials * np.e))
    expected_max_sr = expected_mean_sr + (trials_sr_std * maxZ)
    
    return expected_max_sr, trials_sr_std


def deflated_sharpe_ratio(trials_returns=None, returns_selected=None, expected_mean_sr=0.0, *, expected_max_sr=None):
    """
    Calculate the Deflated Sharpe Ratio (PSR).
    Parameters
    ----------
    trials_returns: pd.DataFrame
        All trials returns, not only the independent trials.
        
    returns_selected: pd.Series
    expected_mean_sr: float
        Expected mean SR, usually 0. We assume that random startegies will have a mean SR of 0,
        expressed in the same frequency as the other parameters.
        
    expected_max_sr: float
        The expected maximum sharpe ratio expected after running all the trials,
        expressed in the same frequency as the other parameters.
    Returns
    -------
    float
    Notes
    -----
    DFS = PSR(SR⁰) = probability that SR^ > SR⁰
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR⁰ = `max_expected_sr`
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
    """
    if expected_max_sr is None:
        expected_max_sr = expected_maximum_sr(trials_returns, expected_mean_sr)
        
    dsr = probabilistic_sharpe_ratio(returns=returns_selected, sr_benchmark=expected_max_sr)

    return dsr

from scipy.stats import skew, kurtosis, sem, gmean, norm
def numerai_sharpe(x):
    return ((np.mean(x) -0) /np.std(x)) #* np.sqrt(12) #-0.010415154

def adj_sharpe(x):
    return numerai_sharpe(x) * (1 + ((skew(x) / 6) * numerai_sharpe(x)) - ((kurtosis(x) - 0) / 24) * (numerai_sharpe(x) ** 2)) #(kurtosis(x) - 3)




def dsr_summary(file_path, prints=False, filter_eras='train_', metric='psr', benchmark=0, scores=[1,0], nrows=None, regime_eras=None):

    era_scores = pd.read_csv('../../Data/processed/era_scores/'+file_path+'_era.csv', nrows=nrows)
    returns_era = era_scores.filter(like=filter_eras, axis=1).T

    mmc_scores = pd.read_csv('../../Data/processed/era_scores/'+file_path+'_mmc.csv', nrows=nrows)
    returns_mmc = mmc_scores.filter(like=filter_eras, axis=1).T

    returns= scores[0]*returns_era + scores[1]*returns_mmc

    if regime_eras !=None:
        returns = returns.T[regime_eras].T

    
    
    if metric=='psr':
        best_psr= probabilistic_sharpe_ratio(returns=returns, sr_benchmark=benchmark).sort_values(ascending=False)
        best_psr_pf_name = best_psr.index[0]
        #print(best_psr)

    else:
        best_psr = returns.apply(lambda x: metric(x), axis=0).sort_values(ascending=False)
        best_psr_pf_name = best_psr.index[0]
        print(str(metric))


    if nrows==1:
        print('PSR: ', best_psr[best_psr_pf_name])
        return returns[best_psr_pf_name]

    best_psr_pf_returns = returns[best_psr_pf_name]
    independent_trials = num_independent_trials(trials_returns=returns)
    exp_max_sr,trials_sr_std = expected_maximum_sr(trials_returns=returns, independent_trials=independent_trials, expected_mean_sr=benchmark)   
    mtr = min_track_record_length(returns)
    dsr = deflated_sharpe_ratio(returns_selected=best_psr_pf_returns, expected_max_sr=exp_max_sr)
    sharpe = numerai_sharpe(best_psr_pf_returns)
    
    if prints==True:
        print('Benchmark: ', benchmark)
        print('Sharpe: ', sharpe)
        #print('best_psr_pf_name: ', best_psr_pf_name)
        print('PSR: ', best_psr[best_psr_pf_name])
        print()
        
        print('independent_trials: ', independent_trials)
        #print('trials_sr_std: ', trials_sr_std)
        print('exp_max_sr: ', exp_max_sr)
        #print('Min Track Lenght: ', mtr[best_psr_pf_name])
        print('DSR: ',dsr)
        print('\nFN Strategy: \n', era_scores.iloc[best_psr_pf_name,:]['hparam'])
        return best_psr_pf_returns

    dict_dsr = {"Metrica": 'Deflated_Sharpe', 
                 "Valor": dsr, 
                 "Categoria": "Special", 
                 "Range_Aceitavel": "[0.5..1]", 
                 "Descricao": "Sharpe Descontado pelas tentativas" }
    
    return dict_dsr






