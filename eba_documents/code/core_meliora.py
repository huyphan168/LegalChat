from scipy.stats import binom
from scipy.stats import norm
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn import metrics
from scipy.stats import t
from scipy import stats
from scipy.stats import ks_2samp


def _binomial(p, d, n):
    """
    A binomial discrete random variable.

    Parameters
    ----------
    p : estimated default probability
    d : number of defaults
    n : number of obligors

    Returns
    -------
    p_value : Binomial test p-value

    Notes
    -----
    If the defaults are modeled as iid Bernoulli trials with success probability p,
    then the number of defaults d is a draw from Binomial(n, p). The one-sided p-value
    is the probability that such a draw would be at least as large as d.
    """

    p_value = 1 - binom.cdf(d - 1, n, p)

    return p_value


def binomial_test(data, ratings, default_flag, predicted_pd, alpha_level=0.05):
    """A binomial test for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns:
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group
        N : number of obligors in each group
        D : number of defaults in each group
        Default Rate : realised default rate per each group
        p_value : Binomial test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Binomial test compares forecasted defaults with observed defaults in a binomial
    model with independent observations under the null hypothesis that the PD applied
    in the portfolio/rating grade at the beginning of the relevant observation period is
    greater than the true one (one-sided hypothesis test). The test statistic is the
    observed number of defaults.

    .. [1] "Studies on the Validation of Internal Rating Systems,"
            Basel Committee on Banking Supervision,
            p. 47, revised May 2005.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import binomial_test
    >>> binomial_test(test_data, 'ratings', 'default_flag', 'probs_default')

               PD    N   D  Default Rate   p_value  reject
    ratings
    A        0.10  401  36      0.089776  0.775347   False
    B        0.15  489  73      0.149284  0.537039   False
    C        0.20  110  23      0.209091  0.443273   False

    """

    # Perform plausibility checks
    if not len(data[ratings].unique()) < 40:
        raise RuntimeError("Number of PD ratings is excessive")
    if not all(x in data.columns for x in [ratings, default_flag, predicted_pd]):
        raise RuntimeError("Missing columns")
    if not all(x in [0, False, 1, True] for x in data[default_flag]):
        raise RuntimeError("Default flag can have only value 0 and 1")
    if not all(x >= 0 and x <= 1 for x in data[predicted_pd]):
        raise RuntimeError("Predicted PDs must be between 0% and 100%")

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean", default_flag: ["count", "sum", "mean"]}).reset_index()
    df.columns = [
        "Rating class",
        "Predicted PD",
        "Total count",
        "Defaults",
        "Actual Default Rate",
    ]

    # Calculate Binomial test outcome for each rating
    df["p_value"] = _binomial(df["Predicted PD"], df["Defaults"], df["Total count"])
    df["Reject H0"] = df["p_value"] < alpha_level

    return df


def _brier(predicted_values, realised_values):
    """

    Parameters
    ----------
    predicted_values : Pandas Series of predicted PD outcomes
    realised_values : Pandas Series of realised PD outcomes

    Returns
    -------
    mse : Brier score for the dataset

    Notes
    -----
    Calculates the mean squared error (MSE) between the outcomes
    and their hypothesized PDs. In this context, the MSE is
    also called the "Brier score" of the dataset.
    """

    # Calculate mean squared error
    errors = realised_values - predicted_values
    mse = (errors**2).sum()

    return mse


def brier_score(data, ratings, default_flag, predicted_pd):
    """Calculate the Brier score for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with at least three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group and total
        N : number of obligors in each group and total
        D : number of defaults in each group and total
        Default Rate : realised default rate per each group and total
        brier_score : overall Brier score


    Notes
    -----
    The Brier score is the mean squared error when each default outcome
    is predicted by its PD rating. Larger values of the Brier score
    indicate a poorer performance of the rating system.

    .. [1] "Studies on the Validation of Internal Rating Systems,"
            Basel Committee on Banking Supervision,
            pp. 46-47, revised May 2005.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import brier_score
    >>> brier_score(test_data, 'ratings', 'default_flag', 'probs_default')

                  PD       N      D  Default Rate brier_score
    ratings
    A        0.10000   401.0   36.0      0.089776        None
    B        0.15000   489.0   73.0      0.149284        None
    C        0.20000   110.0   23.0      0.209091        None
    total    0.13545  1000.0  132.0      0.132000    0.113112

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Not all columns are present"
    assert all(x in [0, False, 1, True] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean", default_flag: ["count", "sum", "mean"]})
    df.columns = ["PD", "N", "D", "Default Rate"]

    # Calculate Brier score for the dataset
    b_score = _brier(df["PD"], df["Default Rate"])

    return b_score


def _herfindahl(df):
    """

    Parameters
    ----------
    df : Pandas DataFrame with first column providing the number
         of obligors and row index corresponding to rating labels

    Returns
    -------
    cv : coefficient of variation
    h : Herfindahl index

    Notes
    -----
    Calculates the coefficient of variation and the Herfindahl index,
    as defined in the paper [1] referenced in herfindahl_test's docstring.
    These quantities measure the dispersion of rating grades in the data.
    """

    k = df.shape[0]
    counts = df.iloc[:, 0]
    n_tot = counts.sum()
    terms = (counts / n_tot - 1 / k) ** 2
    cv = (k * terms.sum()) ** 0.5
    h = (counts**2).sum() / n_tot**2

    return cv, h


def herfindahl_multiple_period_test(data1, data2, ratings, alpha_level=0.05):
    """Calculate the Herfindahl test for a given probability of defaults buckets

    Parameters
    ----------
    data1 : Pandas DataFrame with at least one column
            ratings : PD rating class of obligor
    data2 : Pandas DataFrame with at least one column
            ratings : PD rating class of obligor

    ratings : column label for ratings
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        N_initial : number of obligors in each group and total
        h_initial : Herfindahl index for initial dataset
        N_current : number of obligors in each group and total
        h_current : Herfindahl index for current dataset
        p_value : overall Herfindahl test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Herfindahl test looks for an increase in the
    dispersion of the rating grades over time.
    The (one-sided) null hypothesis is that the current Herfindahl
    index is no greater than the initial Herfindahl index.
    The test statistic is a suitably standardized difference
    in the coefficient of variation, which is monotonically
    related to the Herfindahl index.
    If the Herfindahl index has not changed, then the
    test statistic has the standard Normal distribution.
    Large values of this test statistic
    provide evidence against the null hypothesis.
    (Note that the reference [1] has an uncommon defintion
    of Herfindahl index, whereas we return the common definition)

    .. [1] "Instructions for reporting the validation results
            of internal models - IRB Pillar I models for credit risks," ECB,
            pp. 26-27, 2019.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings1 = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> test_data1 = pd.DataFrame({'ratings': ratings1})
    >>> ratings2 = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> test_data2 = pd.DataFrame({'ratings': ratings2})
    >>> from meliora import herfindahl_test
    >>> herfindahl_test(test_data1, test_data2, "ratings")

           N_initial h_initial  N_current h_current   p_value reject
    B            489      None        487      None      None   None
    A            401      None        414      None      None   None
    C            110      None         99      None      None   None
    total       1000   0.19291       1000  0.206819  0.475327  False

    """

    # Perform plausibility checks
    assert ratings in data1.columns and ratings in data2.columns, f"Ratings column {ratings} not found"
    assert max(len(data1[ratings].unique()), len(data2[ratings].unique())) < 40, "Number of PD ratings is excessive"

    # Transform input data into the required format
    df1 = pd.DataFrame({"N_initial": data1[ratings].value_counts()})
    df2 = pd.DataFrame({"N_current": data2[ratings].value_counts()})

    # Calculate the Herfindahl index for each dataset
    c1, h1 = _herfindahl(df1)
    c2, h2 = _herfindahl(df2)

    # Add a row of totals along with Herfindahl indices
    df1.loc["total"] = [df1["N_initial"].sum()]
    df1["h_initial"] = None
    df1.loc["total", "h_initial"] = h1
    df2.loc["total"] = [df2["N_current"].sum()]
    df2["h_current"] = None
    df2.loc["total", "h_current"] = h2

    # Put the results together into a single dataframe
    df = df1.join(df2)

    # Calculate Herfindahl test's p-value for the dataset
    k = df.shape[0] - 1
    z_stat = (k - 1) ** 0.5 * (c2 - c1) / (c2**2 * (0.5 + c2**2)) ** 0.5
    p_value = 1 - norm.cdf(z_stat)

    # Put the p-value and test result into the output
    df["p_value"] = None
    df.loc["total", "p_value"] = p_value
    if alpha_level:
        df["reject"] = None
        df.loc["total", "reject"] = p_value < alpha_level

    return df


def herfindahl_test(data1, ratings, alpha_level=0.05):
    """Calculate the Herfindahl test for a given probability of defaults buckets

    Parameters
    ----------
    data1 : Pandas DataFrame with at least one column
            ratings : PD rating class of obligor
    data2 : Pandas DataFrame with at least one column
            ratings : PD rating class of obligor

    ratings : column label for ratings
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        N_initial : number of obligors in each group and total
        h_initial : Herfindahl index for initial dataset
        N_current : number of obligors in each group and total
        h_current : Herfindahl index for current dataset
        p_value : overall Herfindahl test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Herfindahl test looks for an increase in the
    dispersion of the rating grades over time.
    The (one-sided) null hypothesis is that the current Herfindahl
    index is no greater than the initial Herfindahl index.
    The test statistic is a suitably standardized difference
    in the coefficient of variation, which is monotonically
    related to the Herfindahl index.
    If the Herfindahl index has not changed, then the
    test statistic has the standard Normal distribution.
    Large values of this test statistic
    provide evidence against the null hypothesis.
    (Note that the reference [1] has an uncommon defintion
    of Herfindahl index, whereas we return the common definition)

    .. [1] "Instructions for reporting the validation results
            of internal models - IRB Pillar I models for credit risks," ECB,
            pp. 26-27, 2019.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings1 = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> test_data1 = pd.DataFrame({'ratings': ratings1})
    >>> ratings2 = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> test_data2 = pd.DataFrame({'ratings': ratings2})
    >>> from meliora import herfindahl_test
    >>> herfindahl_test(test_data1, test_data2, "ratings")

           N_initial h_initial  N_current h_current   p_value reject
    B            489      None        487      None      None   None
    A            401      None        414      None      None   None
    C            110      None         99      None      None   None
    total       1000   0.19291       1000  0.206819  0.475327  False

    """

    # Perform plausibility checks
    assert ratings in data1.columns, f"Ratings column {ratings} not found"
    assert len(data1[ratings].unique()) < 40, "Number of PD ratings is excessive"

    # Transform input data into the required format
    df1 = pd.DataFrame({"N_initial": data1[ratings].value_counts()})

    # Calculate the Herfindahl index for each dataset
    c1, h1 = _herfindahl(df1)

    return c1, h1


def _hosmer(p, d, n):
    """

    Parameters
    ----------
    p : Pandas Series of estimated default probabilities
    d : Pandas Series of number of defaults
    n : Pandas Series of number of obligors

    Returns
    -------
    p_value : Hosmer-Lemeshow Chi-squared test p-value

    Notes
    -----
    Calculates the Hosmer-Lemeshow test statistic, as defined in
    the paper [1] referenced in hosmer_test's docstring.
    If the hypothesized PDs are accurate and defaults are independent,
    this test statisitc is approximately Chi-squared distributed
    with degrees of freedom equal to the number of rating groups minus two.
    The p-value is the probability of such a draw being
    at least as large as the observed value of the statistic.
    """

    assert len(p) > 2, "Hosmer-Lemeshow test requires at least three groups"

    # expected_def = n * p
    # expected_nodef = n * (1 - p)
    # if any(expected_def < 10) or any(expected_nodef < 10):
    #     print("Warning: a group has fewer than 10 expected defaults or non-defaults.")
    #     print("--> Chi-squared approximation is questionable.")

    # terms = (expected_def - d) ** 2 / (p * expected_nodef)
    # chisq_stat = terms.sum()
    # p_value = 1 - chi2.cdf(chisq_stat, len(p) - 2)

    kr = sum((d - p * n) ** 2 / (n * p * (1 - p)))  # todo: treatment of missing values
    p_value = 1 - chi2.cdf(kr, len(p))  # todo: p.val <- pchisq(q = hl, df = k, lower.tail = FALSE)

    return p_value


def hosmer_test(data, ratings, default_flag, predicted_pd, alpha_level=0.05):
    """Calculate the Hosmer-Lemeshow Chi-squared test for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with at least three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group and total
        N : number of obligors in each group and total
        D : number of defaults in each group and total
        Default Rate : realised default rate per each group and total
        p_value : overall Hosmer-Lemeshow test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Hosmer-Lemeshow Chi-squared test calculates a standardized sum
    of squared differences between the number of defaults and
    the expected number of defaults within each rating group.
    Under the null hypothesis that the PDs applied
    in the portfolio/rating grade at the beginning of the relevant observation period are
    equal to the true ones, the test statistic has an approximate Chi-squared distribution.
    Large values of this test statistic
    provide evidence against the null hypothesis.

    .. [1] "Backtesting Framework for PD, EAD and LGD - Public Version,"
            Bauke Maarse, Rabobank International,
            p. 43, 2012.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import hosmer_test
    >>> hosmer_test(test_data, 'ratings', 'default_flag', 'probs_default')

                  PD       N      D  Default Rate   p_value reject
    ratings
    A        0.10000   401.0   36.0      0.089776      None   None
    B        0.15000   489.0   73.0      0.149284      None   None
    C        0.20000   110.0   23.0      0.209091      None   None
    total    0.13545  1000.0  132.0      0.132000  0.468902  False

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Not all columns are present"
    assert all(x in [0, False, 1, True] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean", default_flag: ["count", "sum", "mean"]})
    df.columns = ["PD", "N", "D", "Default Rate"]

    # Calculate Hosmer-Lemeshow test's p-value for the dataset
    p_value = _hosmer(df["PD"], df["D"], df["N"])

    return [p_value, p_value < alpha_level]


def _spiegelhalter(realised_values, predicted_values, alpha_level=0.05):
    """
    todo: https://github.com/andrija-djurovic/PDtoolkit/blob/main/R/12_PREDICTIVE_POWER.R

    Parameters
    ----------
    ratings : Pandas Series of rating categories
    default_flag : Pandas Series of default outcomes (0/1 or False/True)
    df : Pandas DataFrame with ratings as rownames and a column of hypothesized 'PD' values

    Returns
    -------
    p_value : Spiegelhalter test p-value

    Notes
    -----
    Calculates the mean squared error (MSE) between the outcomes
    and their hypothesized PDs, which is approximately Normal.
    If the hypothesized PDs equal the true PDs, then the mean
    and standard deviation of that statistic are provided in
    the paper [1] referenced in spiegelhalter_test's docstring.
    The standardized statistic is approximately standard Normal.
    and leads to a "one-sided" p-value via the Normal cdf.
    """

    # Calculate mean squared error
    errors = realised_values - predicted_values
    mse = (errors**2).sum() / len(errors)

    # # Calculate null expectation and variance of MSE
    expectations = sum(predicted_values * (1 - predicted_values)) / len(realised_values)
    variances = (
        sum(predicted_values * (1 - 2 * predicted_values) ** 2 * (1 - predicted_values)) / len(realised_values) ** 2
    )

    # Calculate standardized statistic
    z_score = (mse - expectations) / np.sqrt(variances)  # todo: check formula

    # Calculate standardized MSE as test statistic, then its p-value
    outcome = z_score > norm.ppf(1 - alpha_level / 2)

    return z_score, outcome


def spiegelhalter_test(data, ratings, default_flag, predicted_pd, alpha_level=0.05):
    """Calculate the Spiegelhalter test for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with at least three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group and total
        N : number of obligors in each group and total
        D : number of defaults in each group and total
        Default Rate : realised default rate per each group and total
        p_value : overall Spiegelhalter test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Spiegelhalter test compares forecasted defaults with observed defaults by analyzing
    the prediction errors. Under the null hypothesis that the PDs applied
    in the portfolio/rating grade at the beginning of the relevant observation period are
    equal to the true ones, the mean squared error can be standardized into
    an approximately standard Normal test statistic. Large values of this test statistic
    provide evidence against the null hypothesis.

    .. [1] "Backtesting Framework for PD, EAD and LGD - Public Version,"
            Bauke Maarse, Rabobank International,
            pp. 43-44, 2012.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import spiegelhalter_test
    >>> spiegelhalter_test(test_data, 'ratings', 'default_flag', 'probs_default')

                  PD       N      D  Default Rate   p_value reject
    ratings
    A        0.10000   401.0   36.0      0.089776      None   None
    B        0.15000   489.0   73.0      0.149284      None   None
    C        0.20000   110.0   23.0      0.209091      None   None
    total    0.13545  1000.0  132.0      0.132000  0.647161  False

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Not all columns are present"
    assert all(x in [0, False, 1, True] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean", default_flag: ["count", "sum", "mean"]})
    df.columns = ["PD", "N", "D", "Default Rate"]

    # Calculate Spiegelhalter test's p-value for the dataset
    result = _spiegelhalter(df["PD"], df["Default Rate"])

    return result


def _jeffreys(p, d, n):
    """

    Parameters
    ----------
    p : estimated default probability
    d : number of defaults
    n : number of obligors

    Returns
    -------
    p_value : Jeffrey's "p-value" (The posterior probability of the null hypothesis)

    Notes
    -----
    Given the Jeffreys prior for the binomial proportion, the
    posterior distribution is a beta distribution with shape parameters a = D + 1/2 and
    b = N − D + 1/2. Here, N is the number of customers in the portfolio/rating grade and
    D is the number of those customers that have defaulted within that observation
    period. The p-value (i.e. the cumulative distribution function of the aforementioned
    beta distribution evaluated at the PD of the portfolio/rating grade) serves as a
    measure of the adequacy of estimated PD.
    """

    a = d + 0.5
    b = n - d + 0.5
    p_value = beta.cdf(p, a, b)

    return p_value


def jeffreys_test(data, ratings, default_flag, predicted_pd, alpha_level=0.05):
    """Calculate the Jeffrey's test for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with at least three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group
        N : number of obligors in each group
        D : number of defaults in each group
        Default Rate : realised default rate per each group
        p_value : Jeffreys p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Jeffreys test compares forecasted defaults with observed defaults in a binomial
    model with independent observations under the null hypothesis that the PD applied
    in the portfolio/rating grade at the beginning of the relevant observation period is
    greater than the true one (one-sided hypothesis test). The test updates a Beta distribution
    (with Jeffrey's prior) in light of the number of defaults and non-defaults,
    then reports the posterior probability of the null hypothesis.

    .. [1] "Instructions for reporting the validation results
            of internal models - IRB Pillar I models for credit risks," ECB,
            pp. 20-21, 2019.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import jeffreys_test
    >>> jeffreys_test(test_data, 'ratings', 'default_flag', 'probs_default')

               PD    N   D  Default Rate   p_value  reject
    ratings
    A        0.10  401  36      0.089776  0.748739   False
    B        0.15  489  73      0.149284  0.511781   False
    C        0.20  110  23      0.209091  0.397158   False

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Not all columns are present"
    assert all(x in [0, False, 1, True] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean", default_flag: ["count", "sum", "mean"]}).reset_index()
    df.columns = [
        "Rating class",
        "Predicted PD",
        "Total count",
        "Defaults",
        "Actual Default Rate",
    ]

    # Calculate Binomial test outcome for each rating
    df["p_value"] = _jeffreys(df["Predicted PD"], df["Defaults"], df["Total count"])
    df["Reject H0"] = df["p_value"] < alpha_level

    return df


def roc_auc(data, target, prediction):
    """Compute Area ROC AUC from prediction scores.

    Note: this implementation can be used with binary, multiclass and
    multilabel classification, but some restrictions apply (see Parameters).
    Read more in the :ref:`User Guide <roc_metrics>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True labels or binary label indicators. The binary and multiclass cases
        expect labels with shape (n_samples,) while the multilabel case expects
        binary label indicators with shape (n_samples, n_classes).
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

    Returns
    -------
    auc : float
        Area Under the Curve score.

    See Also
    --------
    https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/metrics/_ranking.py#L47
    """

    # Perform plausibility checks
    assert all(x >= 0 and x <= 1 for x in data[target]), "Predicted PDs must be between 0% and 100%"
    assert all(x >= 0 and x <= 1 for x in data[prediction]), "Predicted PDs must be between 0% and 100%"

    return roc_auc_score(data[target], data[prediction])


def gini(df, target, prediction):
    """Compute Area ROC AUC from prediction scores.

    todo

    The Ljung-Box (1978) modified portmanteau test. In the
    multivariate time series, this test statistic is asymptotically equal to
    Hosking`. This method and the bottom documentation is taken directly from the
    original 'portes' package.

    The approach used mirrors the one used in Information::create_infotables().

    author Mark Powers <mark.powers@@microsoft.com>

        Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    # Perform plausibility checks
    assert all(x >= 0 and x <= 1 for x in df[target]), "Predicted PDs must be between 0% and 100%"
    assert all(x >= 0 and x <= 1 for x in df[prediction]), "Predicted PDs must be between 0% and 100%"

    roc = roc_auc(df, target, prediction)

    return roc * 2 - 1


def kolmogorov_smirnov_stat(df, target, prediction):
    """Compute Area ROC AUC from prediction scores.

        Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    # Perform plausibility checks
    assert all(x >= 0 and x <= 1 for x in df[target]), "Predicted PDs must be between 0% and 100%"
    assert all(x >= 0 and x <= 1 for x in df[prediction]), "Predicted PDs must be between 0% and 100%"

    result = ks_2samp(df[target], df[prediction])

    return result


def cumulative_lgd_accuracy_ratio(df, predicted_ratings, realised_outcomes):
    """
    CLAR serves as a measure of ranking ability against LGD risk
    The cumulative LGD accuracy ratio (CLAR) curve can be treated as
    the equivalent of the Cumulative Accuracy Profile (CAP) curve. This
    test compares the cumulative percentage of correctly assigned realized
    LGD and the cumulative rate of observations in the predicted LGD bands.

    Parameters
    ----------
    predicted_ratings: pandas Series
        predicted LGD, can be ordinal or continuous
    realised_outcomes: pandas Series
        realised LGD, can be ordinal or continuous

    Returns
    -------
    clar: scalar
        Cumulative LGD Accuracy Ratio

    References
    --------------
    [1] Ozdemir, B., Miu, P., 2009. Basel II Implementation.
    A Guide to Developing and Validating a Compliant Internal Risk Rating
    System. McGraw-Hill, USA.
    [2] See also: https://rdrr.io/cran/VUROCS/man/clar.html

    Examples
    --------
        >>> res = clar(predicted_ratings, realised_outcomes)
        >>> print(res)
    """

    # Calculate CLAR
    x_s = [0]
    x_values = [0]
    y_values = [0]

    for i, j in enumerate(list(set(df[predicted_ratings]))[::-1]):
        x = (df[predicted_ratings] == j).sum()
        x_bucket = df.sort_values(by=realised_outcomes, ascending=False)[x_s[i] : x_s[i] + x]
        x_value = x / len(df)
        y_value = (x_bucket[realised_outcomes] == j).sum() / len((x_bucket[realised_outcomes] == j))
        x_values.append(x_value)
        y_values.append(y_value)
        x_s.append(x + 1)

    new_xvalues = list(np.cumsum(x_values))
    new_yvalues = list(np.cumsum(y_values))

    model_auc = auc(new_xvalues, new_yvalues)
    clar_value = 2 * model_auc

    return clar_value


def loss_capture_ratio(ead, predicted_ratings, realised_outcomes):
    """
    The loss_capture_ratio measures how well a model is able to
    rank LGDs when compared to the observed losses.
    For this approach three plots are relevant: the model loss
    capture curve, ideal loss capture curve and the random loss
    capture curve. These curves are constructed in the same way
    as the curves for the CAP. The main difference is the data,
    which is for LGDs and the LR a (continuous) percentage of the EAD,
    while for the CAP it is binary.
    The LC can be percentage weighted, which simply uses the LGD and
    LR percentages as input, while it can also be EAD weighted, which
    uses the LGD and LR multiplied with the respective EAD as input.
    The results between the two approaches can differ  if the portfolio
    is not-well balanced.

    Parameters
    ----------
    ead: pandas Series
        Exposure at Default
    predicted_ratings: pandas Series
        predicted LGD, can be ordinal or continuous
    realised_outcomes: pandas Series
        realised LGD, can be ordinal or continuous

    Returns
    -------
    LCR: scalar
        Loss Capture Ratio

    References
    ----------------
    Li, D., Bhariok, R., Keenan, S., & Santilli, S. (2009). Validation techniques
    and performance metrics for loss given default models.
    The Journal of Risk Model Validation, 33, 3-26.

    Examples
    --------
        >>> res = loss_capture_ratio(ead, predicted_ratings, realised_outcomes)
        >>> print(res)
    """

    # Create a dataframe
    frame = {
        "ead": ead,
        "predicted_ratings": predicted_ratings,
        "realised_outcomes": realised_outcomes,
    }
    df = pd.DataFrame(frame)

    # Prepare data
    df["loss"] = df["ead"] * df["realised_outcomes"]

    # Model loss capture curve
    df2 = df.sort_values(by="predicted_ratings", ascending=False)
    df2["cumulative_loss"] = df2.cumsum()["loss"]
    df2["cumulative_loss_capture_percentage"] = df2.cumsum()["loss"] / df2.loss.sum()
    auc_curve1 = auc([i for i in range(len(df2))], df2.cumulative_loss_capture_percentage)
    random_auc1 = 0.5 * len(df2) * 1

    # Ideal loss capture curve
    df3 = df.sort_values(by="realised_outcomes", ascending=False)
    df3["cumulative_loss"] = df3.cumsum()["loss"]
    df3["cumulative_loss_capture_percentage"] = df3.cumsum()["loss"] / df3.loss.sum()
    auc_curve2 = auc([i for i in range(len(df3))], df3.cumulative_loss_capture_percentage)
    random_auc2 = 0.5 * len(df3) * 1

    lcr = (auc_curve1 - random_auc1) / (auc_curve2 - random_auc2)

    return lcr


def bayesian_error_rate(df, default_flag, prob_default):
    """
    BER is the proportion of the whole sample that is misclassified
    when the rating system is in optimal use. For a perfect rating model,
    the BER has a value of zero. A model's BER depends on the probability
    of default. The lower the BER, and the lower the classification error,
    the better the model.

    The Bayesian error rate specifies the minimum probability of error if
    the rating system or score function under consideration is used for a
    yes/no decision whether a borrower will default or not. The error can
    be estimated parametrically, e.g. assuming noFrmal score distributions,
    or non-parametrically, for instance with kernel density estimation methods.
    If parametric estimation is applied, the distributional assumptions have
    to be carefully checked. Non-parametric estimation will be critical if
    sample sizes are small. In its general form, the error rate depends on
    the total portfolio probability of default. As a consequence, in many
    cases its magnitude is influenced much more by the probability of
    erroneously identifying a non-defaulter as a defaulter than by the
    probability of not detecting a defaulter.
    In practice, therefore, the error rate is often applied
    with a fictitious 50% probability of default. In this case, the error
    rate is equivalent to the Kolmogorov-Smirnov statistic and to the Pietra index.

    Parameters
    ----------
    default_flag : pandas series
        Boolean flag indicating whether the borrower has actually defaulted
    prob_default : pandas series
        Predicted default probability, as returned by a classifier.

    Returns
    ---------
    score : float
        Bayesian Error Rate.

    Examples
    --------
    >>> from scipy import stats
    >>> default_flag = [1, 0, 0, 1, 1]
    >>> prob_default = [0.01, 0.04, 0.07, 0.11, 0]
    >>> bayesian_error_rate(default_flag, prob_default)
    -0.47140452079103173
    """

    # frame = {"default_flag": default_flag, "prob_default": prob_default}

    # df = pd.DataFrame(frame)

    fpr, tpr, thresholds = metrics.roc_curve(df[default_flag], df[prob_default])
    roc_curve_df = pd.DataFrame({"c": thresholds, "hit_rate": tpr, "false_alarm_rate": fpr})

    p_d = df.default_flag.sum() / len(df)

    roc_curve_df["ber"] = p_d * (1 - roc_curve_df.hit_rate) + (1 - p_d) * roc_curve_df.false_alarm_rate

    return round(min(roc_curve_df["ber"]), 3)


def information_value(df, feature, target, pr=0):
    """
    A numerical value that quantifies the predictive power of an independent
    variable in capturing the binary dependent variable.
    Weight of evidence (WOE) is a measure of how much the evidence supports or
    undermines a hypothesis. WOE measures the relative risk of an attribute of
    binning level. The value depends on whether the value of the target variable
    is a nonevent or an event.
    The information value (IV) is a weighted sum of the WOE of the
    characteristic's attributes. The weight is the difference between the
    conditional probability of an attribute for an event and the conditional
    probability of that attribute for a nonevent.
    An information value can be any real number. Generally speaking, the higher
    the information value, the more predictive an attribute is likely to be.
    Parameters
    ----------
    df : Pandas dataframe
        Contains information on the the feature and target variable
    feature : string
        independent variable
    feature : string
        dependent variable
    Returns
    -------
    iv : float
       Information Value.
    References
    --------------
    -  https://www.lexjansen.com/mwsug/2013/AA/MWSUG-2013-AA14.pdf.
    -  https://documentation.sas.com/doc/en/vdmmlcdc/8.1/casstat/viyastat_binning_details02.htm.
    Examples
    --------
    >>> iv = calc_iv(df, feature, target, pr=0)
    >>> iv
    -0.47140452079103173

    """

    lst = []

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append(
            [
                feature,
                val,
                df[df[feature] == val].count()[feature],
                df[(df[feature] == val) & (df[target] == 1)].count()[feature],
            ]
        )

    data = pd.DataFrame(lst, columns=["Variable", "Value", "All", "Bad"])
    data = data[data["Bad"] > 0]

    data["Share"] = data["All"] / data["All"].sum()
    data["Bad Rate"] = data["Bad"] / data["All"]
    data["Distribution Good"] = (data["All"] - data["Bad"]) / (data["All"].sum() - data["Bad"].sum())
    data["Distribution Bad"] = data["Bad"] / data["Bad"].sum()
    data["WoE"] = np.log(data["Distribution Good"] / data["Distribution Bad"])
    data["IV"] = data["WoE"] * (data["Distribution Good"] - data["Distribution Bad"])

    data = data.sort_values(by=["Variable", "Value"], ascending=True)
    iv = data["IV"].sum()

    return data, iv


def lgd_t_test(df, observed_lgd, expected_lgd, level="portfolio", segment_col=None):
    """t-test for the Null hypothesis that estimated LGD is greater than true LGD
    Parameters
    ----------
    df: array-like, at least 2D
        data
    observed_LGD_col: string
        name of column with observed LGD values
    expected_LGD_col: string
        name of column with expected LGD values
    level: string
        'portfolio' (single comparison) or 'segment' level (multiple comparisons)
    verbose: boolean
        if true, results and interpretation are printed
    Returns
    -------
    N: integer
        Number of customers
    LGD.mean: float
        Mean value of observed LGD values
    pred_LGD.mean: float
        Mean value of predicted LGD values
    t_stat: float
        test statistic
    lgd_s2: float
        denominator of test statistic
    p_value: float
        p-value of the test
    Notes
    -----------
    Observations are assumed to be independent.
    This fundtion can be used for both performing and non-performing LGDs.
    Examples
    --------
    .. code-block:: python
        >>> res = lgd_t_test(df=df, observed_LGD_col='LGD', expected_LGD_col='PRED_LGD', verbose=True)
        >>> print(res)
    """
    # Checking for any missing data
    if df.empty:
        raise TypeError("No data provided!")
    if observed_lgd is None:
        raise TypeError("No column name for observed LGDs provided")
    if expected_lgd is None:
        raise TypeError("No column name for expected LGDs provided")

    # Check the data for missing values
    if df[observed_lgd].hasnans:
        raise ValueError("Missing values in {}".format(observed_lgd))
    if df[expected_lgd].hasnans:
        raise ValueError("Missing values in {}".format(expected_lgd))

    results = []
    if level == "pool":

        for segment in df[segment_col].unique():
            df_segment = df[df[segment_col] == segment]

            length = len(df_segment)
            obs_lgd = df_segment[observed_lgd]
            pred_lgd = df_segment[expected_lgd]
            error = obs_lgd - pred_lgd
            mean_error = error.mean()
            num = np.sqrt(length) * mean_error
            lgd_s2 = ((error - mean_error) ** 2).sum() / (length - 1)
            t_stat = num / np.sqrt(lgd_s2)
            p_value = 1 - t.cdf(t_stat, df=length - 1)

            results.append(
                [
                    segment,
                    length,
                    obs_lgd.mean(),
                    pred_lgd.mean(),
                    lgd_s2,
                    mean_error,
                    t_stat,
                    p_value,
                ]
            )

    else:
        segment = "None"
        length = len(df)
        obs_lgd = df[observed_lgd]
        pred_lgd = df[expected_lgd]
        error = obs_lgd - pred_lgd
        mean_error = error.mean()
        num = np.sqrt(length) * mean_error
        lgd_s2 = ((error - mean_error) ** 2).sum() / (length - 1)
        t_stat = num / np.sqrt(lgd_s2)
        p_value = 1 - t.cdf(t_stat, df=length - 1)

        results.append(
            [
                segment,
                length,
                obs_lgd.mean(),
                pred_lgd.mean(),
                lgd_s2,
                mean_error,
                t_stat,
                p_value,
            ]
        )

        # from list of lists to dataframe
    results_df = pd.DataFrame(
        results,
        columns=[
            "segment",
            "N",
            "realised_lgd_mean",
            "pred_lgd_mean",
            "s2",
            "mean_error",
            "t_stat",
            "p_value",
        ],
    )

    return results_df  # todo: results do not make sense


def migration_matrix_stability(df, initial_ratings_col, final_ratings_col):
    """z-tests to verify stability of transition matrices
    Parameters
    ----------
    df: array-like, at least 2D
        data
    initial_ratings_col: string
        name of column with initial ratings values
    final_ratings_col: string
        name of column with final ratings values
    Returns
    -------
    z_df: array-like
        z statistic for each ratings pair
    phi_df: array-like
        p-values for each ratings pair
    Notes
    -----------
    The Null hypothesis is that p_ij >= p_ij-1 or p_ij-1 >= p_ij
    depending on whether the (ij) entry is below or above main diagonal
    Examples
    --------
    .. code-block:: python
        >>> res = migration_matrix_stability(df=df, initial_ratings_col='ratings', final_ratings_col='ratings2')
        >>> print(res)
    """
    a = df[initial_ratings_col]
    b = df[final_ratings_col]
    N_ij = pd.crosstab(a, b)
    p_ij = pd.crosstab(a, b, normalize="index")
    K = len(set(a))
    z_df = p_ij.copy()
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            if i == j:

                z_ij = np.nan

            if i > j:
                Ni = N_ij.sum(axis=1).values[i - 1]

                num = p_ij.iloc[i - 1, j - 1 + 1] - p_ij.iloc[i - 1, j - 1]
                den_a = p_ij.iloc[i - 1, j - 1] * (1 - p_ij.iloc[i - 1, j - 1]) / Ni
                den_b = p_ij.iloc[i - 1, j - 1 + 1] * (1 - p_ij.iloc[i - 1, j - 1 + 1]) / Ni
                den_c = 2 * p_ij.iloc[i - 1, j - 1] * p_ij.iloc[i - 1, j - 1 + 1] / Ni

                z_ij = num / np.sqrt(den_a + den_b + den_c)

            elif i < j:
                Ni = N_ij.sum(axis=1).values[i - 1]

                num = p_ij.iloc[i - 1, j - 1 - 1] - p_ij.iloc[i - 1, j - 1]
                den_a = p_ij.iloc[i - 1, j - 1] * (1 - p_ij.iloc[i - 1, j - 1]) / Ni
                den_b = p_ij.iloc[i - 1, j - 1 - 1] * (1 - p_ij.iloc[i - 1, j - 1 - 1]) / Ni
                den_c = 2 * p_ij.iloc[i - 1, j - 1] * p_ij.iloc[i - 1, j - 1 - 1] / Ni

                z_ij = num / np.sqrt(den_a + den_b + den_c)

            else:

                z_ij = np.nan

            z_df.iloc[i - 1, j - 1] = z_ij
    phi_df = z_df.apply(lambda x: x.apply(norm.cdf))
    return z_df, phi_df


def population_stability_index(data, bin_flag, variable):  # todo: expected vs actual
    """Calculate the PSI for a single variable

    Args:
        expected_array: numpy array of original values
        actual_array: numpy array of new values, same size as expected
        buckets: number of percentile ranges to bucket the values into
    Returns:
        psi_value: calculated PSI value

        Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    df = pd.crosstab(data[variable], data[bin_flag], normalize="columns")
    df.columns = ["actual", "expected"]

    df["expected"] = np.where(df["expected"] == 0, 0.0001, df["expected"])

    # Calculating PSI
    df["PSI"] = (df["actual"] - df["expected"]) * np.log(df["actual"] / df["expected"])

    psi = np.sum(df["PSI"])

    return df, psi


def kendall_tau(x, y, variant="b"):
    """
    Calculate Kendall's tau, a correlation measure for ordinal data.
    This is a wrapper around SciPy kendalltau function.
    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.
    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.
    variant: {'b', 'c'}, optional
        Defines which variant of Kendall's tau is returned. Default is 'b'.
    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.
    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.
    [2] Maurice G. Kendall, "The treatment of ties in ranking problems",
           Biometrika Vol. 33, No. 3, pp. 239-251. 1945.
    [3] Gottfried E. Noether, "Elements of Nonparametric Statistics",
        John Wiley & Sons, 1967.
    [4] Peter M. Fenwick, "A new data structure for cumulative frequency tables",
        Software: Practice and Experience, Vol. 24, No. 3, pp. 327-336, 1994.
    [5] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),
           Charles Griffin & Co., 1970.
    Scipy: https://github.com/scipy/scipy/blob/v1.8.1/scipy/stats/_stats_py.py#L4666-L4875
    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> x2 = [1, 4, 7, 1, 0]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    >>> p_value
    0.2827454599327748
    """

    tau, pvalue = stats.kendalltau(x, y, initial_lexsort=None, variant="b")

    return tau, pvalue


def somersd(array_1, array_2, alternative="two-sided"):
    """
    Calculates Somers' D, an asymmetric measure of ordinal association.
    This is a wrapper around scipy.stats.somersd function.
    Somers' :math:`D` is a measure of the correspondence between two rankings.
    It considers the difference between the number of concordant
    and discordant pairs in two rankings and is  normalized such that values
    close  to 1 indicate strong agreement and values close to -1 indicate
    strong disagreement.
    Parameters
    ----------
    x: array_like
        1D array of rankings, treated as the (row) independent variable.
        Alternatively, a 2D contingency table.
    y: array_like, optional
        If `x` is a 1D array of rankings, `y` is a 1D array of rankings of the
        same length, treated as the (column) dependent variable.
        If `x` is 2D, `y` is ignored.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:
        * 'two-sided': the rank correlation is nonzero
        * 'less': the rank correlation is negative (less than zero)
        * 'greater':  the rank correlation is positive (greater than zero)
    Returns
    -------
    res : SomersDResult
        A `SomersDResult` object with the following fields:
            correlation : float
               The Somers' :math:`D` statistic.
            pvalue : float
               The p-value for a hypothesis test whose null
               hypothesis is an absence of association, :math:`D=0`.
               See notes for more information.
            table : 2D array
               The contingency table formed from rankings `x` and `y` (or the
               provided contingency table, if `x` is a 2D array)
    References
    ----------
    [1] Robert H. Somers, "A New Asymmetric Measure of Association for
           Ordinal Variables", *American Sociological Review*, Vol. 27, No. 6,
           pp. 799--811, 1962.
    [2] Morton B. Brown and Jacqueline K. Benedetti, "Sampling Behavior of
           Tests for Correlation in Two-Way Contingency Tables", *Journal of
           the American Statistical Association* Vol. 72, No. 358, pp.
           309--315, 1977.
    [3] SAS Institute, Inc., "The FREQ Procedure (Book Excerpt)",
           *SAS/STAT 9.2 User's Guide, Second Edition*, SAS Publishing, 2009.
    [4] Laerd Statistics, "Somers' d using SPSS Statistics", *SPSS
           Statistics Tutorials and Statistical Guides*,
           https://statistics.laerd.com/spss-tutorials/somers-d-using-spss-statistics.php,
           Accessed July 31, 2020.
    Examples
    --------
    >>> table = [[27, 25, 14, 7, 0], [7, 14, 18, 35, 12], [1, 3, 2, 7, 17]]
    >>> res = somersd(table)
    >>> res.statistic
    0.6032766111513396
    >>> res.pvalue
    1.0007091191074533e-27

    """

    return stats.somersd(array_1, array_2, alternative="two-sided")


def spearman_correlation(array_1, array_2):
    """
    Calculate a Spearman correlation coefficient with associated p-value.
    This is a wrapper around scipy.stats.spearmanr function.
    The Spearman rank-order correlation coefficient is a nonparametric
    measure of the monotonicity of the relationship between two datasets.
    Unlike the Pearson correlation, the Spearman correlation does not
    assume that both datasets are normally distributed. Like other
    correlation coefficients, this one varies between -1 and +1 with 0
    implying no correlation. Correlations of -1 or +1 imply an exact
    monotonic relationship. Positive correlations imply that as x
    increases, so does y. Negative correlations imply that as x increases,
    y decreases.
    The p-value roughly indicates the probability of an uncorrelated
    system producing datasets that have a Spearman correlation at least
    as extreme as the one computed from these datasets. The p-values
    are not entirely reliable but are probably reasonable for datasets
    larger than 500 or so.

    Parameters
    ----------
    array_1 : pandas series
        Series containing multiple observations
    array_2 : pandas series
        Series containing multiple observations
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:
        * 'two-sided': the correlation is nonzero
        * 'less': the correlation is negative (less than zero)
        * 'greater':  the correlation is positive (greater than zero)

    Returns
    -------
    correlation : float or ndarray (2-D square)
        Spearman correlation matrix or correlation coefficient (if only 2
        variables are given as parameters. Correlation matrix is square with
        length equal to total number of variables (columns or rows) in ``a``
        and ``b`` combined.
    pvalue : float
        The p-value for a hypothesis test whose null hypotheisis
        is that two sets of data are uncorrelated. See `alternative` above
        for alternative hypotheses. `pvalue` has the same
        shape as `correlation`.

    References
    -------------
        [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
        Probability and Statistics Tables and Formulae. Chapman & Hall: New
        York. 2000.
        Section  14.7

    Examples
    --------
    >>> spearmanr([1,2,3,4,5], [5,6,7,8,7])
    SpearmanrResult(correlation=0.82078..., pvalue=0.08858...)
    """

    return stats.spearmanr(array_1, array_2, alternative="two-sided")


def pearson_correlation(array_1, array_2):
    """
    Calculate a Pearson correlation coefficient with associated p-value.
    This is a wrapper around scipy.stats.pearsonr function.
    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear
    relationship. Positive correlations imply that as x increases, so does
    y. Negative correlations imply that as x increases, y decreases.
    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.

    Parameters
    ----------
    array_1 : pandas series
        Series containing multiple observations
    array_2 : pandas series
        Series containing multiple observations

    Returns
    -------
    correlation : float
        Pearson's correlation coefficient.
    pvalue : float
        Two-tailed p-value.

    References
    ----------
    [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
        Probability and Statistics Tables and Formulae. Chapman & Hall: New
    """

    return stats.spearmanr(array_1, array_2)


def migration_matrices_statistics(df, period_1_ratings, period_2_ratings):
    """
    The objective of this validation tool is to analyse the migration of customers across
    rating grades during the relevant observation period

        Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    a = df[period_1_ratings]
    b = df[period_2_ratings]

    n_ij = pd.crosstab(a, b)
    p_ij = pd.crosstab(a, b, normalize="index")

    mnormu = 0
    K = len(set(df["period_1_ratings"]))
    for i in range(1, K - 1 + 1):
        for j in range(i + 1, K + 1):
            cac = p_ij.iloc[i - 1 : i, i:].sum(axis=1).values[0]
            b = n_ij.sum(axis=1).values[i - 1]
            a = max(i - K, i - 1)
            mnormu += a * b * cac

    mnorml = 0
    K = len(set(df["period_1_ratings"]))
    for i in range(2, K + 1):
        for j in range(1, i - 1 + 1):
            coc = p_ij.iloc[i - 1 : i, i:].sum(axis=1).values[0]
            b = n_ij.sum(axis=1).values[i - 1]  # todo
            a = max(i - K, i - 1)
            mnorml += a * b * coc

    upper_mwb = 0
    for i in range(1, K - 1 + 1):
        for j in range(i + 1, K + 1):
            upper_mwb += abs(i - j) * n_ij.sum(axis=1).values[i - 1] * p_ij.iloc[i - 1, j - 1]
    upper_mwb = (1 / mnormu) * upper_mwb

    lower_mwb = 0
    for i in range(2, K + 1):
        for j in range(1, i - 1 + 1):
            lower_mwb += abs(i - j) * n_ij.sum(axis=1).values[i - 1] * p_ij.iloc[i - 1, j - 1]
    lower_mwb = (1 / mnorml) * lower_mwb

    return upper_mwb, lower_mwb


def _entropy(data, realised_pd, count):
    """
    CIER measures the ratio of distance between Unconditional and Conditional Entropy to
    Unconditional Entropy.

        Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    # prepare data
    data["perc"] = data[count] / data[count].sum()

    # unconditional entropy
    pd = (data[count] * data[realised_pd]).sum() / data[count].sum()

    # Unconditional entropy
    h0 = -(pd * np.log(pd) + (1 - pd) * np.log(1 - pd))

    # Conditional entropy
    data["hc"] = -(
        data[realised_pd] * np.log(data[realised_pd]) + (1 - data[realised_pd]) * np.log(1 - data[realised_pd])
    )
    h1 = sum(data["perc"] * data["hc"])

    return h0, h1


def conditional_information_entropy_ratio(data, realised_pd, count):
    """CIER measures the ratio of distance between Unconditional and
    Conditional Entropy to Unconditional Entropy.

        Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    h0, h1 = _entropy(data, realised_pd, count)

    return (h0 - h1) / h0


def kullback_leibler_dist(data, realised_pd, count):
    """CIER measures the ratio of distance between Unconditional and
    Conditional Entropy to Unconditional Entropy."""

    h0, h1 = _entropy(data, realised_pd, count)

    return h0 - h1


def loss_shortfall(data, ead, predicted_lgd, realised_lgd):
    """
    Loss Shortfall is a measure of the difference between the expected loss and the
    realised loss. It is a measure of the loss that would have been incurred if the
    actual losses were equal to the expected losses.

    Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    estimated_losses = data[ead] * data[predicted_lgd]
    realised_losses = data[ead] * data[realised_lgd]

    return 1 - estimated_losses.sum() / realised_losses.sum()


def mean_absolute_deviation(data, ead, predicted_lgd, realised_lgd):
    """
    Mean Absolute Deviation is a measure of the difference between the expected loss and
    the realised loss. It is a measure of the loss that would have been incurred if the
    actual losses were equal to the expected losses.

        Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    return np.sum(np.abs(data[realised_lgd] - data[predicted_lgd]) * data[ead]) / data[ead].sum()


def elbe_t_test(df, lgd, elbe):
    """
    # df should contain the facilities for which backtesting will be performed.
    # LGD is the observed LGD, ELBE is the ELBE for each facility, dataframe
    # columns should be defined in this manner

        Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    N = len(df)
    error = df[lgd] - df[elbe]
    mean_error = error.mean()
    num = np.sqrt(N) * mean_error
    s2 = (((df[lgd] - df[elbe]) - mean_error) ** 2).sum() / (N - 1)
    t_stat = num / np.sqrt(s2)
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=N - 1))

    return pd.DataFrame({"facilities": [N], "lgd_mean": [df[elbe].mean()], "t_stat": [t_stat], "p_value": [p_value]})


def normal_test(predicted_pd, realised_pd, alpha=0.05):
    """
    - Numeric vector of calibrated probabilities of default (PD).
    - Numeric vector of observed default rates.
    - alpha Significance level of p-value for implemented tests

    The Normality test is a test of the null hypothesis that the residuals of a
    regression are normally distributed.

    The normal test is an approach to deal with the dependence problem that occurs in
    the case of the binomial and chi-square tests. The normal test is a multi-period
    test of correctness of a default probability forecast for a single rating category.

    It is applied under the assumption that the mean default rate does not vary too
    much over time and that default events in different years are independent. The
    normal test is motivated by the Central Limit Theorem and is based on a normal
    approximation of the distribution of the time-averaged default rates.

    Simulation studies show that the quality of the normal approximation is moderate
    but exhibits a conservative bias. As a consequence, the true Type I error tends
    to be lower than the nominal level of the test, i.e. the proportion of
    erroneous rejections of PD forecasts will be smaller than might be expected from
    the formal confidence level of the test. The test seems even to be, to a certain
    degree, robust against a violation of the assumption that defaults are independent
    over time. However, the power of the test is moderate, in particular for short
    time series (for example five years).

        Calculate Kendall's tau, a correlation measure for ordinal data.

    This method and the bottom documentation is taken directly from the
    original SciPy package (kendalltau).

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    """

    length = len(predicted_pd)

    error = predicted_pd - realised_pd

    standard_error = sum((error) ** 2) - ((sum(error) ** 2 / length)) / (length - 1)
    t_stat = sum(predicted_pd - realised_pd) / np.sqrt(standard_error * length)
    p_value = t.cdf(abs(t_stat), df=length)
    estimate = sum(predicted_pd - realised_pd)

    # create dataframe from inputs
    return pd.DataFrame({"estimate": [estimate], "t_stat": [t_stat], "p_value": [p_value], "outcome": [np.nan]})


def redelmeier_test(df):
    """Redelmeier test for the equality of two binomial proportions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'DEFAULT_FLAG' and 'ADJUSTED_PD'.

    Returns
    -------
    z : float
        The test statistic.
    p_value : float
        The p-value for a hypothesis test whose null hypothesis is
        an absence of association, tau = 0.

    References
    ----------
    [1] Redelmeier, D. A. (1996). A test for the equality of two binomial proportions.
        Statistics in medicine, 15(1), 1-11.

    Examples
    --------
    >>> from meliora import redelmeier_test
    >>> x1 = [0.1, 0.2, 0.3, 0.4, 0.5]  # todo
    >>> x2 = [0.1, 0.3, 0.3, 0.4, 0.1]
    >>> z, p_value = redelmeier_test(x1, x2)
    >>> z
    -0.47140452079103173

    """
    pd1 = df["ADJUSTED_PD"]
    pd2 = df["min_PD"]
    y = df["DEFAULT_FLAG"]

    t1 = pd1 - pd2
    t2 = pd1 + pd2

    z = ((pd1**2 - pd2**2) - 2 * t1 * y).sum() / ((t1**2 * t2 * (2 - t1)).sum()) ** 0.5

    p_value = norm.sf(abs(z)) * 2

    if p_value < 0.05:
        print("reject")
    else:
        print("not reject")

    return z, p_value