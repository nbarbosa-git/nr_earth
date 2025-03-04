# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
This module creates clustered subsets of features described in the paper Clustered Feature Importance (Presentation
Slides) by Dr. Marcos Lopez de Prado. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595 and is also explained
in the book Machine Learning for Asset Managers Snippet 6.5.2 page 84.
"""
# pylint: disable=invalid-name

# Imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.regression.linear_model import OLS

#from mlfinlab.clustering.onc import get_onc_clusters
#from mlfinlab.codependence.codependence_matrix import get_dependence_matrix, get_distance_matrix
#from mlfinlab.util import devadarsh


try:
    from mlfinlab.clustering.onc import get_onc_clusters
    from mlfinlab.codependence.codependence_matrix import get_dependence_matrix, get_distance_matrix


except:
    from onc import get_onc_clusters
    from codependence_matrix import get_dependence_matrix, get_distance_matrix


def get_feature_clusters(X: pd.DataFrame, Xcorr: pd.DataFrame, dependence_metric: str, distance_metric: str = None,
                         linkage_method: str = None, n_clusters: int = None, check_silhouette_scores: bool = False,
                         critical_threshold: float = 0.0) -> list:
    """
    Machine Learning for Asset Managers
    Snippet 6.5.2.1 , page 85. Step 1: Features Clustering

    Gets clustered features subsets from the given set of features.

    :param X: (pd.DataFrame) Dataframe of features.
    :param Xcorr: (pd.DataFrame) Dataframe of Correlation.
    :param dependence_metric: (str) Method to be use for generating dependence_matrix, either 'linear' or
                              'information_variation' or 'mutual_information' or 'distance_correlation'.
    :param distance_metric: (str) The distance operator to be used for generating the distance matrix. The methods that
                            can be applied are: 'angular', 'squared_angular', 'absolute_angular'. Set it to None if the
                            feature are to be generated as it is by the ONC algorithm.
    :param linkage_method: (str) Method of linkage to be used for clustering. Methods include: 'single', 'ward',
                           'complete', 'average', 'weighted', and 'centroid'. Set it to None if the feature are to
                           be generated as it is by the ONC algorithm.
    :param n_clusters: (int) Number of clusters to form. Must be less the total number of features. If None then it
                       returns optimal number of clusters decided by the ONC Algorithm.
    :param check_silhouette_scores: (bool) Flag to check if X contains features with low silh. scores and modify it.
    :param critical_threshold: (float) Threshold for determining low silhouette score in the dataset. It can any real number
                                in [-1,+1], default is 0 which means any feature that has a silhouette score below 0 will be
                                indentified as having low silhouette and hence required transformation will be appiled to for
                                for correction of the same.
    :return: (list) Feature subsets.
    """

    #devadarsh.track('get_feature_clusters')

    # Get the dependence matrix
    if dependence_metric != 'linear':
        dep_matrix = get_dependence_matrix(X, dependence_method=dependence_metric)

    else:
        #dep_matrix = X.corr()
        dep_matrix = Xcorr



    # Checking if dataset contains features low silhouette
    if check_silhouette_scores is True:
        X = _check_for_low_silhouette_scores(X, dep_matrix, critical_threshold)

    if n_clusters is None and (distance_metric is None or linkage_method is None):
        return list(get_onc_clusters(dep_matrix.fillna(0))[1].values())  # Get optimal number of clusters

    if distance_metric is not None and (linkage_method is not None and n_clusters is None):
        n_clusters = len(get_onc_clusters(dep_matrix.fillna(0))[1])
        
    if n_clusters >= len(X.columns):  # Check if number of clusters exceeds number of features
        raise ValueError('Number of clusters must be less than the number of features')

    # Apply distance operator on the dependence matrix
    dist_matrix = get_distance_matrix(dep_matrix, distance_metric=distance_metric)

    # Get the linkage
    link = linkage(squareform(dist_matrix), method=linkage_method)
    clusters = fcluster(link, t=n_clusters, criterion='maxclust')
    clustered_subsets = [[f for c, f in zip(clusters, X.columns) if c == ci] for ci in range(1, n_clusters + 1)]

    return clustered_subsets


def _cluster_transformation(X: pd.DataFrame, clusters: dict, feats_to_transform: list) -> pd.DataFrame:
    """
    Machine Learning for Asset Managers
    Snippet 6.5.2.1 , page 85. Step 1: Features Clustering (last paragraph)

    Transforms a dataset to reduce the multicollinearity of the system by replacing the original feature with
    the residual from regression.

    :param X: (pd.DataFrame) Dataframe of features.
    :param clusters: (dict) Clusters generated by ONC algorithm.
    :param feats_to_transform: (list) Features that have low silhouette score and to be transformed.
    :return: (pd.DataFrame) Transformed features.
    """

    for feat in feats_to_transform:
        for i, j in clusters.items():

            if feat in j:  # Selecting the cluster that contains the feature
                exog = sm.add_constant(X.drop(j, axis=1)).values
                endog = X[feat].values
                ols = OLS(endog, exog).fit()

                if ols.df_model < (exog.shape[1] - 1):
                    # Degree of freedom is low
                    new_exog = _combine_features(X, clusters, i)
                    # Run the regression again on the new exog
                    ols = OLS(endog, new_exog.reshape(exog.shape[0], -1)).fit()
                    X[feat] = ols.resid
                else:
                    X[feat] = ols.resid

    return X


def _combine_features(X, clusters, exclude_key) -> np.array:
    """
    Combines features of each cluster linearly by following a minimum variance weighting scheme.
    The Minimum Variance weights are calculated without constraints, other than the weights sum to one.

    :param X: (pd.DataFrame) Dataframe of features.
    :param clusters: (dict) Clusters generated by ONC algorithm.
    :param exclude_key: (int) Key of the cluster which is to be excluded.
    :return: (np.array) Combined features for each cluster.
    """

    new_exog = []
    for i, cluster in clusters.items():

        if i != exclude_key:
            subset = X[cluster]
            cov_matx = subset.cov()  # Covariance matrix of the cluster
            eye_vec = np.array(cov_matx.shape[1] * [1], float)
            try:
                numerator = np.dot(np.linalg.inv(cov_matx), eye_vec)
                denominator = np.dot(eye_vec, numerator)
                # Minimum variance weighting
                wghts = numerator / denominator
            except np.linalg.LinAlgError:
                # A singular matrix so giving each component equal weight
                wghts = np.ones(subset.shape[1]) * (1 / subset.shape[1])
            new_exog.append(((subset * wghts).sum(1)).values)

    return np.array(new_exog)


def _check_for_low_silhouette_scores(X: pd.DataFrame, dep_matrix: pd.DataFrame,
                                     critical_threshold: float = 0.0) -> pd.DataFrame:
    """
    Machine Learning for Asset Managers
    Snippet 6.5.2.1 , page 85. Step 1: Features Clustering (last paragraph)

    Checks where the dataset contains features low silhouette due one feature being a combination of
    multiple features across clusters. This is a problem, because ONC cannot assign one feature to multiple
    clusters and it needs a transformation.

    :param X: (pd.DataFrame) Dataframe of features.
    :param dep_matrix: (pd.DataFrame) Dataframe with dependences between features.
    :param critical_threshold: (float) Threshold for determining low silhouette score.
    :return: (pd.DataFrame) Dataframe of features.
    """

    _, clstrs, silh = get_onc_clusters(dep_matrix)
    low_silh_feat = silh[silh < critical_threshold].index
    if len(low_silh_feat) > 0:
        print(f'{len(low_silh_feat)} feature/s found with low silhouette score {low_silh_feat}. Returning the transformed dataset')

        # Returning the transformed dataset
        return _cluster_transformation(X, clstrs, low_silh_feat)

    print('No feature/s found with low silhouette score. All features belongs to its respective clusters')

    return X
