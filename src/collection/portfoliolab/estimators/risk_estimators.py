# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/portfoliolab/blob/master/LICENSE.txt

# pylint: disable=missing-module-docstring
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.covariance import MinCovDet, EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS
from scipy.optimize import minimize
from scipy.cluster.hierarchy import average, complete, single, dendrogram
from matplotlib import pyplot as plt

#from portfoliolab.estimators.returns_estimators import ReturnsEstimators
#from portfoliolab.utils import devadarsh



try:
    from portfoliolab.estimators.returns_estimators import ReturnsEstimators


except:

    from returns_estimators import ReturnsEstimators






class RiskEstimators:
    """
    This class contains the implementations for different ways to calculate and adjust Covariance matrices.
    The functions related to de-noising and de-toning the Covariance matrix are reproduced with modification
    from Chapter 2 of the the following book:
    Marcos Lopez de Prado “Machine Learning for Asset Managers”, (2020).
    """

    def __init__(self):
        """
        Initialize
        """
        #devadarsh.track('RiskEstimators')

    @staticmethod
    def minimum_covariance_determinant(returns, price_data=False, assume_centered=False,
                                       support_fraction=None, random_state=None):
        """
        Calculates the Minimum Covariance Determinant for a dataframe of asset prices or returns.

        This function is a wrap of the sklearn's MinCovDet (MCD) class. According to the
        scikit-learn User Guide on Covariance estimation:

        "The idea is to find a given proportion (h) of “good” observations that are not outliers
        and compute their empirical covariance matrix. This empirical covariance matrix is then
        rescaled to compensate for the performed selection of observations".

        Link to the documentation:
        <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html>`_

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param assume_centered: (bool) Flag for data with mean significantly equal to zero.
                                       (Read the documentation for MinCovDet class, False by default)
        :param support_fraction: (float) Values between 0 and 1. The proportion of points to be included in the support
                                         of the raw MCD estimate. (Read the documentation for MinCovDet class,
                                         None by default)
        :param random_state: (int) Seed used by the random number generator. (None by default)
        :return: (np.array) Estimated robust covariance matrix.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimators()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Calculating the covariance matrix
        cov_matrix = MinCovDet(assume_centered=assume_centered, support_fraction=support_fraction,
                               random_state=random_state).fit(returns).covariance_

        return cov_matrix

    @staticmethod
    def empirical_covariance(returns, price_data=False, assume_centered=False):
        """
        Calculates the Maximum likelihood covariance estimator for a dataframe of asset prices or returns.

        This function is a wrap of the sklearn's EmpiricalCovariance class. According to the
        scikit-learn User Guide on Covariance estimation:

        "The covariance matrix of a data set is known to be well approximated by the classical maximum
        likelihood estimator, provided the number of observations is large enough compared to the number
        of features (the variables describing the observations). More precisely, the Maximum Likelihood
        Estimator of a sample is an unbiased estimator of the corresponding population’s covariance matrix".

        Link to the documentation:
        <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html>`_

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param assume_centered: (bool) Flag for data with mean almost, but not exactly zero.
                                       (Read documentation for EmpiricalCovariance class, False by default)
        :return: (np.array) Estimated covariance matrix.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimators()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Calculating the covariance matrix
        cov_matrix = EmpiricalCovariance(assume_centered=assume_centered).fit(returns).covariance_

        return cov_matrix

    @staticmethod
    def shrinked_covariance(returns, price_data=False, shrinkage_type='basic', assume_centered=False,
                            basic_shrinkage=0.1):
        """
        Calculates the Covariance estimator with shrinkage for a dataframe of asset prices or returns.

        This function allows three types of shrinkage - Basic, Ledoit-Wolf and Oracle Approximating Shrinkage.
        It is a wrap of the sklearn's ShrunkCovariance, LedoitWolf and OAS classes. According to the
        scikit-learn User Guide on Covariance estimation:

        "Sometimes, it even occurs that the empirical covariance matrix cannot be inverted for numerical
        reasons. To avoid such an inversion problem, a transformation of the empirical covariance matrix
        has been introduced: the shrinkage. Mathematically, this shrinkage consists in reducing the ratio
        between the smallest and the largest eigenvalues of the empirical covariance matrix".

        Link to the documentation:
        <https://scikit-learn.org/stable/modules/covariance.html>`_

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param shrinkage_type: (str) Type of shrinkage to use. (``basic`` by default, ``lw``, ``oas``, ``all``)
        :param assume_centered: (bool) Flag for data with mean almost, but not exactly zero.
                                       (Read documentation for chosen shrinkage class, False by default)
        :param basic_shrinkage: (float) Between 0 and 1. Coefficient in the convex combination for basic shrinkage.
                                        (0.1 by default)
        :return: (np.array) Estimated covariance matrix. Tuple of covariance matrices if shrinkage_type = ``all``.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimators()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Calculating the covariance matrix for the chosen method
        if shrinkage_type == 'basic':
            cov_matrix = ShrunkCovariance(assume_centered=assume_centered, shrinkage=basic_shrinkage).fit(
                returns).covariance_
        elif shrinkage_type == 'lw':
            cov_matrix = LedoitWolf(assume_centered=assume_centered).fit(returns).covariance_
        elif shrinkage_type == 'oas':
            cov_matrix = OAS(assume_centered=assume_centered).fit(returns).covariance_
        else:
            cov_matrix = (
                ShrunkCovariance(assume_centered=assume_centered, shrinkage=basic_shrinkage).fit(returns).covariance_,
                LedoitWolf(assume_centered=assume_centered).fit(returns).covariance_,
                OAS(assume_centered=assume_centered).fit(returns).covariance_)

        return cov_matrix

    @staticmethod
    def semi_covariance(returns, price_data=False, threshold_return=0):
        """
        Calculates the Semi-Covariance matrix for a dataframe of asset prices or returns.

        Semi-Covariance matrix is used to calculate the portfolio's downside volatility. Usually, the
        threshold return is zero and the negative volatility is measured. A threshold can be a positive number
        when one assumes a required return rate. If the threshold is above zero, the output is the volatility
        measure for returns below this threshold.

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param threshold_return: (float) Required return for each period in the frequency of the input data.
                                         (If the input data is daily, it's a daily threshold return, 0 by default)
        :return: (np.array) Semi-Covariance matrix.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimators()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Returns that are lower than the threshold
        lower_returns = returns - threshold_return < 0

        # Calculating the minimum of 0 and returns minus threshold
        min_returns = (returns - threshold_return) * lower_returns

        # Simple covariance matrix
        semi_covariance = returns.cov()

        # Iterating to fill elements
        for row_number in range(semi_covariance.shape[0]):
            for column_number in range(semi_covariance.shape[1]):
                # Series of returns for the element from the row and column
                row_asset = min_returns.iloc[:, row_number]
                column_asset = min_returns.iloc[:, column_number]

                # Series of element-wise products
                covariance_series = row_asset * column_asset

                # Element of the Semi-Covariance matrix
                semi_cov_element = covariance_series.sum() / min_returns.size

                # Inserting the element in the Semi-Covariance matrix
                semi_covariance.iloc[row_number, column_number] = semi_cov_element

        return semi_covariance

    @staticmethod
    def exponential_covariance(returns, price_data=False, window_span=60):
        """
        Calculates the Exponentially-weighted Covariance matrix for a dataframe of asset prices or returns.

        It calculates the series of covariances between elements and then gets the last value of exponentially
        weighted moving average series from covariance series as an element in matrix.

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param window_span: (int) Used to specify decay in terms of span for the exponentially-weighted series.
                                  (60 by default)
        :return: (np.array) Exponentially-weighted Covariance matrix.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimators()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Simple covariance matrix
        cov_matrix = returns.cov()

        # Iterating to fill elements
        for row_number in range(cov_matrix.shape[0]):
            for column_number in range(cov_matrix.shape[1]):
                # Series of returns for the element from the row and column
                row_asset = returns.iloc[:, row_number]
                column_asset = returns.iloc[:, column_number]

                # Series of covariance
                covariance_series = (row_asset - row_asset.mean()) * (column_asset - column_asset.mean())

                # Exponentially weighted moving average series
                ew_ma = covariance_series.ewm(span=window_span).mean()

                # Using the most current element as the Exponential Covariance value
                cov_matrix.iloc[row_number, column_number] = ew_ma[-1]

        return cov_matrix

    @staticmethod
    def filter_corr_hierarchical(cor_matrix, method='complete', draw_plot=False):
        """
        Creates a filtered correlation matrix using hierarchical clustering methods from an empirical
        correlation matrix, given that all values are non-negative [0 ~ 1]

        This function allows for three types of hierarchical clustering - complete, single, and average
        linkage clusters. Link to hierarchical clustering methods documentation:
        `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_

        It works as follows:

        First, the method creates a hierarchical clustering tree using scipy's hierarchical clustering methods
        from the empirical 2-D correlation matrix.

        Second, it extracts and stores each cluster's filtered value (alpha) and assigns it to it's corresponding leaf.

        Finally, we create a new filtered matrix by assigning each of the correlations to their corresponding
        parent node's alpha value.

        :param cor_matrix: (np.array) Numpy array of an empirical correlation matrix.
        :param method: (str) Hierarchical clustering method to use. (``complete`` by default, ``single``, ``average``)
        :param draw_plot: (bool) Plots the hierarchical cluster tree. (False by default)
        :return: (np.array) The filtered correlation matrix.
        """

        # Check if all matrix elements are positive
        if np.any(cor_matrix < 0):
            warnings.warn('Not all elements in matrix are positive... Returning unfiltered matrix.', UserWarning)
            return cor_matrix

        # Check if matrix is 2-D
        if len(cor_matrix.shape) == 2:
            cor_x, cor_y = cor_matrix.shape
        else:
            warnings.warn('Invalid matrix dimensions, input must be 2-D array... Returning unfiltered matrix.', UserWarning)
            return cor_matrix

        # Check if matrix dimensions and diagonal values are valid.
        if cor_x == cor_y and np.allclose(np.diag(cor_matrix), 1): # using np.allclose as diag values might be 0.99999
            # Creating new coorelation condensed matrix for the upper triangle and dismissing the diagnol.
            new_cor = cor_matrix[np.triu_indices(cor_matrix.shape[0], k=1)]
        else:
            warnings.warn('Invalid matrix, input must be a correlation matrix of size (m x m)... Returning unfiltered matrix.', UserWarning)
            return cor_matrix

        # Compute the hierarchical clustering tree
        if method == 'complete':
            z_cluster = complete(new_cor)
        elif method == 'single':
            z_cluster = single(new_cor)
        elif method == 'average':
            z_cluster = average(new_cor)
        else:
            warnings.warn('Invalid method selected, please check docstring... Returning unfiltered matrix.', UserWarning)
            return cor_matrix

        # Plot the hierarchical cluster tree
        if draw_plot:
            fig = plt.figure(figsize=(10, 6))
            axis = fig.add_subplot(111)
            dendrogram(z_cluster, ax=axis)
            plt.show()

        # Creates a pd.DataFrame that will act as a dictionary where the index is the leaf node id, and the values are
        # thier corresponding cluster's alpha value
        alpha_values = z_cluster[:, 2]
        alphas = z_cluster[:, 0]
        df_alphas = pd.DataFrame(alpha_values, index=alphas)
        df_alphas.loc[z_cluster[0][1]] = alpha_values[0]

        # Creates the filtered correlation matrix
        alphas_sorterd = df_alphas.sort_index()
        alphas_x = np.tile(alphas_sorterd.values, (1, len(alphas_sorterd.values)))
        filt_corr = np.maximum(alphas_x, alphas_x.T)
        np.fill_diagonal(filt_corr, 1)

        return filt_corr

    def denoise_covariance(self, cov, tn_relation, denoise_method='const_resid_eigen', detone=False,
                           market_component=1, kde_bwidth=0.01, alpha=0):
        """
        De-noises the covariance matrix or the correlation matrix.

        Two denoising methods are supported:
        1. Constant Residual Eigenvalue Method (``const_resid_eigen``)
        2. Spectral Method (``spectral``)
        3. Targeted Shrinkage Method (``target_shrink``)


        The Constant Residual Eigenvalue Method works as follows:

        First, a correlation is calculated from the covariance matrix (if the input is the covariance matrix).

        Second, eigenvalues and eigenvectors of the correlation matrix are calculated using the linalg.eigh
        function from numpy package.

        Third, a maximum theoretical eigenvalue is found by fitting Marcenko-Pastur (M-P) distribution
        to the empirical distribution of the correlation matrix eigenvalues. The empirical distribution
        is obtained through kernel density estimation using the KernelDensity class from sklearn.
        The fit of the M-P distribution is done by minimizing the Sum of Squared estimate of Errors
        between the theoretical pdf and the kernel. The minimization is done by adjusting the variation
        of the M-P distribution.

        Fourth, the eigenvalues of the correlation matrix are sorted and the eigenvalues lower than
        the maximum theoretical eigenvalue are set to their average value. This is how the eigenvalues
        associated with noise are shrinked. The de-noised covariance matrix is then calculated back
        from new eigenvalues and eigenvectors.

        The Spectral Method works just like the Constant Residual Eigenvalue Method, but instead of replacing
        eigenvalues lower than the maximum theoretical eigenvalue to their average value, they are replaced with
        zero instead.

        The Targeted Shrinkage Method works as follows:

        First, a correlation is calculated from the covariance matrix (if the input is the covariance matrix).

        Second, eigenvalues and eigenvectors of the correlation matrix are calculated using the linalg.eigh
        function from numpy package.

        Third, the correlation matrix composed from eigenvectors and eigenvalues related to noise is
        shrunk using the alpha variable. The shrinkage is done by summing the noise correlation matrix
        multiplied by alpha to the diagonal of the noise correlation matrix multiplied by (1-alpha).

        Fourth, the shrinked noise correlation matrix is summed to the information correlation matrix.

        Correlation matrix can also be detoned by excluding a number of first eigenvectors representing
        the market component.

        These algorithms are reproduced with minor modifications from the following book:
        Marcos Lopez de Prado “Machine Learning for Asset Managers”, (2020).

        :param cov: (np.array) Covariance matrix or correlation matrix.
        :param tn_relation: (float) Relation of sample length T to the number of variables N used to calculate the
                                    covariance matrix.
        :param denoise_method: (str) Denoising methos to use. (``const_resid_eigen`` by default, ``target_shrink``)
        :param detone: (bool) Flag to detone the matrix. (False by default)
        :param market_component: (int) Number of fist eigevectors related to a market component. (1 by default)
        :param kde_bwidth: (float) The bandwidth of the kernel to fit KDE.
        :param alpha: (float) In range (0 to 1) - shrinkage of the noise correlation matrix to use in the
                              Targeted Shrinkage Method. (0 by default)
        :return: (np.array) De-noised covariance matrix or correlation matrix.
        """

        # Correlation matrix computation (if correlation matrix given, nothing changes)

        corr = self.cov_to_corr(cov)

        # Calculating eigenvalues and eigenvectors
        eigenval, eigenvec = self._get_pca(corr)

        # Calculating the maximum eigenvalue to fit the theoretical distribution
        maximum_eigen, _ = self._find_max_eval(np.diag(eigenval), tn_relation, kde_bwidth)

        # Calculating the threshold of eigenvalues that fit the theoretical distribution
        # from our set of eigenvalues
        num_facts = eigenval.shape[0] - np.diag(eigenval)[::-1].searchsorted(maximum_eigen)

        if denoise_method == 'target_shrink':
            # Based on the threshold, de-noising the correlation matrix
            corr = self._denoised_corr_targ_shrink(eigenval, eigenvec, num_facts, alpha)
        elif denoise_method == 'spectral':
            # Based on the threshold, de-noising the correlation matrix
            corr = self._denoised_corr_spectral(eigenval, eigenvec, num_facts)
        else: # Default const_resid_eigen method
            # Based on the threshold, de-noising the correlation matrix
            corr = self._denoised_corr(eigenval, eigenvec, num_facts)

        # Detone the correlation matrix if needed
        if detone:
            corr = self._detoned_corr(corr, market_component)

        # Calculating the covariance matrix from the de-noised correlation matrix
        cov_denoised = self.corr_to_cov(corr, np.diag(cov) ** (1 / 2))

        return cov_denoised

    @staticmethod
    def corr_to_cov(corr, std):
        """
        Recovers the covariance matrix from a correlation matrix.

        Requires a vector of standard deviations of variables - square root
        of elements on the main diagonal fo the covariance matrix.

        Formula used: Cov = Corr * OuterProduct(std, std)

        :param corr: (np.array) Correlation matrix.
        :param std: (np.array) Vector of standard deviations.
        :return: (np.array) Covariance matrix.
        """

        cov = corr * np.outer(std, std)
        return cov

    @staticmethod
    def cov_to_corr(cov):
        """
        Derives the correlation matrix from a covariance matrix.

        Formula used: Corr = Cov / OuterProduct(std, std)

        :param cov: (np.array) Covariance matrix.
        :return: (np.array) Covariance matrix.
        """

        # Calculating standard deviations of the elements
        std = np.sqrt(np.diag(cov))

        # Transforming to correlation matrix
        corr = cov / np.outer(std, std)

        # Making sure correlation coefficients are in (-1, 1) range
        corr[corr < -1], corr[corr > 1] = -1, 1

        return corr

    @staticmethod
    def is_matrix_invertible(matrix):
        """
        Check if a matrix is invertible or not.

        :param matrix: (Numpy matrix) A matrix whose invertibility we want to check.
        :return: (bool) Boolean value depending on whether the matrix is invertible or not.
        """

        return matrix.shape[0] == matrix.shape[1] and np.linalg.matrix_rank(matrix) == matrix.shape[0]

    @staticmethod
    def _fit_kde(observations, kde_bwidth=0.01, kde_kernel='gaussian', eval_points=None):
        """
        Fits kernel to a series of observations (in out case eigenvalues), and derives the
        probability density function of observations.

        The function used to fit kernel is KernelDensity from sklearn.neighbors. Fit of the KDE
        can be evaluated on a given set of points, passed as eval_points variable.

        :param observations: (np.array) Array of observations (eigenvalues) eigenvalues to fit kernel to.
        :param kde_bwidth: (float) The bandwidth of the kernel. (0.01 by default)
        :param kde_kernel: (str) Kernel to use [``gaussian`` by default, ``tophat``, ``epanechnikov``, ``exponential``,
                                 ``linear``,``cosine``].
        :param eval_points: (np.array) Array of values on which the fit of the KDE will be evaluated.
                                       If None, the unique values of observations are used. (None by default)
        :return: (pd.Series) Series with estimated pdf values in the eval_points.
        """

        # Reshaping array to a vertical one
        observations = observations.reshape(-1, 1)

        # Estimating Kernel Density of the empirical distribution of eigenvalues
        kde = KernelDensity(kernel=kde_kernel, bandwidth=kde_bwidth).fit(observations)

        # If no specific values provided, the fit KDE will be valued on unique eigenvalues.
        if eval_points is None:
            eval_points = np.unique(observations).reshape(-1, 1)

        # If the input vector is one-dimensional, reshaping to a vertical one
        if len(eval_points.shape) == 1:
            eval_points = eval_points.reshape(-1, 1)

        # Evaluating the log density model on the given values
        log_prob = kde.score_samples(eval_points)

        # Preparing the output of pdf values
        pdf = pd.Series(np.exp(log_prob), index=eval_points.flatten())

        return pdf

    @staticmethod
    def _mp_pdf(var, tn_relation, num_points):
        """
        Derives the pdf of the Marcenko-Pastur distribution.

        Outputs the pdf for num_points between the minimum and maximum expected eigenvalues.
        Requires the variance of the distribution (var) and the relation of T - the number
        of observations of each X variable to N - the number of X variables (T/N).

        :param var: (float) Variance of the M-P distribution.
        :param tn_relation: (float) Relation of sample length T to the number of variables N (T/N).
        :param num_points: (int) Number of points to estimate pdf.
        :return: (pd.Series) Series of M-P pdf values.
        """

        # Changing the type as scipy.optimize.minimize outputs np.array with one element to this function
        if not isinstance(var, float):
            var = float(var)

        # Minimum and maximum expected eigenvalues
        eigen_min = var * (1 - (1 / tn_relation) ** (1 / 2)) ** 2
        eigen_max = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2

        # Space of eigenvalues
        eigen_space = np.linspace(eigen_min, eigen_max, num_points)

        # Marcenko-Pastur probability density function for eigen_space
        pdf = tn_relation * ((eigen_max - eigen_space) * (eigen_space - eigen_min)) ** (1 / 2) / \
                             (2 * np.pi * var * eigen_space)
        pdf = pd.Series(pdf, index=eigen_space)

        return pdf

    def _pdf_fit(self, var, eigen_observations, tn_relation, kde_bwidth, num_points=1000):
        """
        Calculates the fit (Sum of Squared estimate of Errors) of the empirical pdf
        (kernel density estimation) to the theoretical pdf (Marcenko-Pastur distribution).

        SSE is calculated for num_points, equally spread between minimum and maximum
        expected theoretical eigenvalues.

        :param var: (float) Variance of the M-P distribution. (for the theoretical pdf)
        :param eigen_observations: (np.array) Observed empirical eigenvalues. (for the empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N. (for the theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel. (for the empirical pdf)
        :param num_points: (int) Number of points to estimate pdf. (for the empirical pdf, 1000 by default)
        :return: (float) SSE between empirical pdf and theoretical pdf.
        """

        # Calculating theoretical and empirical pdf
        theoretical_pdf = self._mp_pdf(var, tn_relation, num_points)
        empirical_pdf = self._fit_kde(eigen_observations, kde_bwidth, eval_points=theoretical_pdf.index.values)

        # Fit calculation
        sse = np.sum((empirical_pdf - theoretical_pdf) ** 2)

        return sse

    def _find_max_eval(self, eigen_observations, tn_relation, kde_bwidth):
        """
        Searching for maximum random eigenvalue by fitting Marcenko-Pastur distribution
        to the empirical one - obtained through kernel density estimation. The fit is done by
        minimizing the Sum of Squared estimate of Errors between the theoretical pdf and the
        kernel fit. The minimization is done by adjusting the variation of the M-P distribution.

        :param eigen_observations: (np.array) Observed empirical eigenvalues. (for the empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N. (for the theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel. (for the empirical pdf)
        :return: (float, float) Maximum random eigenvalue, optimal variation of the Marcenko-Pastur distribution.
        """

        # Searching for the variation of Marcenko-Pastur distribution for the best fit with the empirical distribution
        optimization = minimize(self._pdf_fit, x0=np.array(0.5), args=(eigen_observations, tn_relation, kde_bwidth),
                                bounds=((1e-5, 1 - 1e-5),))

        # The optimal solution found
        var = optimization['x'][0]

        # Eigenvalue calculated as the maximum expected eigenvalue based on the input
        maximum_eigen = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2

        return maximum_eigen, var

    @staticmethod
    def _get_pca(hermit_matrix):
        """
        Calculates eigenvalues and eigenvectors from a Hermitian matrix. In our case, from the correlation matrix.

        Function used to calculate the eigenvalues and eigenvectors is linalg.eigh from numpy package.

        Eigenvalues in the output are placed on the main diagonal of a matrix.

        :param hermit_matrix: (np.array) Hermitian matrix.
        :return: (np.array, np.array) Eigenvalues matrix, eigenvectors array.
        """

        # Calculating eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(hermit_matrix)

        # Index to sort eigenvalues in descending order
        indices = eigenvalues.argsort()[::-1]

        # Sorting
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # Outputting eigenvalues on the main diagonal of a matrix
        eigenvalues = np.diagflat(eigenvalues)

        return eigenvalues, eigenvectors

    def _denoised_corr(self, eigenvalues, eigenvectors, num_facts):
        """
        De-noises the correlation matrix using the Constant Residual Eigenvalue method.

        The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
        of the first eigenvalue that is below the maximum theoretical eigenvalue.

        De-noising is done by shrinking the eigenvalues associated with noise (the eigenvalues lower than
        the maximum theoretical eigenvalue are set to a constant eigenvalue, preserving the trace of the
        correlation matrix).

        The result is the de-noised correlation matrix.

        :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
        :param eigenvectors: (float) Eigenvectors array.
        :param num_facts: (float) Threshold for eigenvalues to be fixed.
        :return: (np.array) De-noised correlation matrix.
        """

        # Vector of eigenvalues from the main diagonal of a matrix
        eigenval_vec = np.diag(eigenvalues).copy()

        # Replacing eigenvalues after num_facts to their average value
        eigenval_vec[num_facts:] = eigenval_vec[num_facts:].sum() / float(eigenval_vec.shape[0] - num_facts)

        # Back to eigenvalues on main diagonal of a matrix
        eigenvalues = np.diag(eigenval_vec)

        # De-noised correlation matrix
        corr = np.dot(eigenvectors, eigenvalues).dot(eigenvectors.T)

        # Rescaling the correlation matrix to have 1s on the main diagonal
        corr = self.cov_to_corr(corr)

        return corr

    def _denoised_corr_spectral(self, eigenvalues, eigenvectors, num_facts):
        """
        De-noises the correlation matrix using the Spectral method.

        The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
        of the first eigenvalue that is below the maximum theoretical eigenvalue.

        De-noising is done by shrinking the eigenvalues associated with noise (the eigenvalues lower than
        the maximum theoretical eigenvalue are set to zero, preserving the trace of the
        correlation matrix).
        The result is the de-noised correlation matrix.

        :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
        :param eigenvectors: (float) Eigenvectors array.
        :param num_facts: (float) Threshold for eigenvalues to be fixed.
        :return: (np.array) De-noised correlation matrix.
        """

        # Vector of eigenvalues from the main diagonal of a matrix
        eigenval_vec = np.diag(eigenvalues).copy()

        # Replacing eigenvalues after num_facts to zero
        eigenval_vec[num_facts:] = 0

        # Back to eigenvalues on main diagonal of a matrix
        eigenvalues = np.diag(eigenval_vec)

         # De-noised correlation matrix
        corr = np.dot(eigenvectors, eigenvalues).dot(eigenvectors.T)

        # Rescaling the correlation matrix to have 1s on the main diagonal
        corr = self.cov_to_corr(corr)

        return corr

    @staticmethod
    def _denoised_corr_targ_shrink(eigenvalues, eigenvectors, num_facts, alpha=0):
        """
        De-noises the correlation matrix using the Targeted Shrinkage method.

        The input is the correlation matrix, the eigenvalues and the eigenvectors of the correlation
        matrix and the number of the first eigenvalue that is below the maximum theoretical eigenvalue
        and the shrinkage coefficient for the eigenvectors and eigenvalues associated with noise.

        Shrinks strictly the random eigenvalues - eigenvalues below the maximum theoretical eigenvalue.

        The result is the de-noised correlation matrix.

        :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
        :param eigenvectors: (float) Eigenvectors array.
        :param num_facts: (float) Threshold for eigenvalues to be fixed.
        :param alpha: (float) In range (0 to 1) - shrinkage among the eigenvectors.
                              and eigenvalues associated with noise. (0 by default)
        :return: (np.array) De-noised correlation matrix.
        """

        # Getting the eigenvalues and eigenvectors related to signal
        eigenvalues_signal = eigenvalues[:num_facts, :num_facts]
        eigenvectors_signal = eigenvectors[:, :num_facts]

        # Getting the eigenvalues and eigenvectors related to noise
        eigenvalues_noise = eigenvalues[num_facts:, num_facts:]
        eigenvectors_noise = eigenvectors[:, num_facts:]

        # Calculating the correlation matrix from eigenvalues associated with signal
        corr_signal = np.dot(eigenvectors_signal, eigenvalues_signal).dot(eigenvectors_signal.T)

        # Calculating the correlation matrix from eigenvalues associated with noise
        corr_noise = np.dot(eigenvectors_noise, eigenvalues_noise).dot(eigenvectors_noise.T)

        # Calculating the De-noised correlation matrix
        corr = corr_signal + alpha * corr_noise + (1 - alpha) * np.diag(np.diag(corr_noise))

        return corr

    def _detoned_corr(self, corr, market_component=1):
        """
        De-tones the correlation matrix by removing the market component.

        The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
        of the first eigenvalue that is above the maximum theoretical eigenvalue and the number of
        eigenvectors related to a market component.

        :param corr: (np.array) Correlation matrix to detone.
        :param market_component: (int) Number of fist eigevectors related to a market component. (1 by default)
        :return: (np.array) De-toned correlation matrix.
        """

        # Calculating eigenvalues and eigenvectors of the de-noised matrix
        eigenvalues, eigenvectors = self._get_pca(corr)

        # Getting the eigenvalues and eigenvectors related to market component
        eigenvalues_mark = eigenvalues[:market_component, :market_component]
        eigenvectors_mark = eigenvectors[:, :market_component]

        # Calculating the market component correlation
        corr_mark = np.dot(eigenvectors_mark, eigenvalues_mark).dot(eigenvectors_mark.T)

        # Removing the market component from the de-noised correlation matrix
        corr = corr - corr_mark

        # Rescaling the correlation matrix to have 1s on the main diagonal
        corr = self.cov_to_corr(corr)

        return corr
