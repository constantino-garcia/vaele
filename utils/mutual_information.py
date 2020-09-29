import numpy as np
import scipy.stats as stats
from sklearn.metrics import mutual_info_score


def freedman_diaconis(data):
    """ Use Freedman Diaconis rule to compute optimal histogram bin width. """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N    = data.size
    bw = (2 * IQR) / np.power(N, 1/3)
    datmin, datmax = np.min(data), np.max(data)
    datrng = datmax - datmin
    return int((datrng / bw) + 1)


def calc_ami(x, max_lag=None, nbins=None):
    def calc_mi(x, y, nbins):
        c_xy = np.histogram2d(x, y, nbins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi

    max_lag = int(0.5 * len(x)) if max_lag is None else max_lag

    # nbins = freedman_diaconis(x) if nbins is None else nbins
    nbins = max(int(len(x) ** (1 / 3)), 2) if nbins is None else nbins
    max_x = max(x)
    bins = np.linspace(min(x) - 1e-6, max_x, nbins)
    if max_x not in bins:
        np.concatenate([bins, np.array([max_x + 1e-6])])

    ami_results = np.zeros(max_lag)
    ami_results[0] = calc_mi(x, x, bins)
    for lag in range(1, max_lag):
        ami_results[lag] = calc_mi(x[lag:], x[:-lag], bins)

    return ami_results


def estimate_lag_value(x):
    """Searches for first e-decay in the AMI of the x signal"""
    ami = calc_ami(x)
    indices = np.where(ami < ami[0] * np.exp(-1))[0]
    if len(indices) > 0:
        return indices[0]
    else:
        return np.nan

