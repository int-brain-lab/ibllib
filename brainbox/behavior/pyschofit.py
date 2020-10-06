"""
The psychofit toolbox contains tools to fit two-alternative psychometric
data. The fitting is done using maximal likelihood estimation: one
assumes that the responses of the subject are given by a binomial
distribution whose mean is given by the psychometric function.
The data can be expressed in fraction correct (from .5 to 1) or in
fraction of one specific choice (from 0 to 1). To fit them you can use
these functions:
  weibull50          - Weibull function from 0.5 to 1, with lapse rate
  weibull            - Weibull function from 0 to 1, with lapse rate
  erf_psycho         - erf function from 0 to 1, with lapse rate
  erf_psycho_2gammas - erf function from 0 to 1, with two lapse rates
Functions in the toolbox are:
  mle_fit_psycho     - Maximumum likelihood fit of psychometric function
  neg_likelihood     - Negative likelihood of a psychometric function
For more info, see:
  Examples           - Examples of use of psychofit toolbox
Matteo Carandini, 2000-2015
"""

import functools
import numpy as np
import scipy.optimize
from scipy.special import erf


def mle_fit_psycho(data, P_model='weibull', parstart=None, parmin=None, parmax=None, nfits=5):
    """
    Maximumum likelihood fit of psychometric function.
    Args:
        data: 3 x n matrix where first row corresponds to stim levels,
              the second to number of trials for each stim level (int),
              the third to proportion correct / proportion rightward (float between 0 and 1)
        P_model: The psychometric function. Possibilities include 'weibull'
                 (DEFAULT), 'weibull50', 'erf_psycho' and 'erf_psycho_2gammas'
        parstart: Non-zero starting parameters, used to try to avoid local minima.
                  The parameters are [threshold, slope, gamma], or if using the
                  'erf_psycho_2gammas' model append a second gamma value.
                  Recommended to use a value > 1. If None, some reasonable defaults are used.
        parmin: Minimum parameter values.  If None, some reasonable defaults are used
        parmax: Maximum parameter values.  If None, some reasonable defaults are used
        nfits: The number of fits
    Returns:
        pars: The parameters from the best of the fits
        L: The likelihood of the best fit
    Raises:
        TypeError: data must be a list or numpy array
        ValueError: data must be m by 3 matrix
    Examples:
        Below we fit a Weibull function to some data:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> cc = np.array([-8., -6., -4., -2.,  0.,  2.,  4.,  6.,  8.]) # contrasts
        >>> nn = np.full((9,), 10) # number of trials at each contrast
        >>> pp = np.array([5., 8., 20., 41., 54., 59., 79., 92., 96])/100 # proportion "rightward"
        >>> pars, L = mle_fit_psycho(np.vstack((cc, nn, pp)), 'erf_psycho')
        >>> plt.plot(cc, pp, 'bo', mfc='b')
        >>> plt.plot(np.arange(-8, 8, 0.1), erf_psycho(pars, np.arange(-8, 8, 0.1)), '-b')
    Information:
        1999-11 FH wrote it
        2000-01 MC cleaned it up
        2000-04 MC took care of the 50% case
        2009-12 MC replaced fmins with fminsearch
        2010-02 MC, AZ added nfits
        2013-02 MC+MD fixed bug with dealing with NaNs
        2018-08 MW ported to Python
    """
    # Input validation
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError('data must be a list or numpy array')

    if data.shape[0] != 3:
        raise ValueError('data must be m by 3 matrix')

    rep = lambda x: (x, x) if P_model.endswith('2gammas') else (x,) # noqa
    if parstart is None:
        parstart = np.array([np.mean(data[0, :]), 3., *rep(.05)])
    if parmin is None:
        parmin = np.array([np.min(data[0, :]), 0., *rep(0.)])
    if parmax is None:
        parmax = np.array([np.max(data[0, :]), 10., *rep(.4)])

    # find the good values in pp (conditions that were effectively run)
    ii = np.isfinite(data[2, :])

    likelihoods = np.zeros(nfits,)
    pars = np.empty((nfits, parstart.size))

    f = functools.partial(neg_likelihood, data=data[:, ii],
                          P_model=P_model, parmin=parmin, parmax=parmax)
    for ifit in range(nfits):
        pars[ifit, :] = scipy.optimize.fmin(f, parstart, disp=False)
        parstart = parmin + np.random.rand(parmin.size) * (parmax - parmin)
        likelihoods[ifit] = -neg_likelihood(pars[ifit, :], data[:, ii], P_model, parmin, parmax)

    # the values to be output
    L = likelihoods.max()
    iBestFit = likelihoods.argmax()
    return pars[iBestFit, :], L


def neg_likelihood(pars, data, P_model='weibull', parmin=None, parmax=None):
    """
    Negative likelihood of a psychometric function.
    Args:
        pars: Model parameters [threshold, slope, gamma], or if
              using the 'erf_psycho_2gammas' model append a second gamma value.
        data: 3 x n matrix where first row corresponds to stim levels,
              the second to number of trials for each stim level (int),
              the third to proportion correct / proportion rightward (float between 0 and 1)
        P_model: The psychometric function. Possibilities include 'weibull'
                 (DEFAULT), 'weibull50', 'erf_psycho' and 'erf_psycho_2gammas'
        parmin: Minimum bound for parameters.  If None, some reasonable defaults are used
        parmax: Maximum bound for parameters.  If None, some reasonable defaults are used
    Returns:
        ll: The likelihood of the parameters.  The equation is:
            - sum(nn.*(pp.*log10(P_model)+(1-pp).*log10(1-P_model)))
            See the the appendix of Watson, A.B. (1979). Probability
            summation over time. Vision Res 19, 515-522.
    Raises:
        ValueError: invalid model, options are "weibull",
                    "weibull50", "erf_psycho" and "erf_psycho_2gammas"
        TypeError: data must be a list or numpy array
        ValueError data must be m by 3 matrix
    Information:
        1999-11 FH wrote it
        2000-01 MC cleaned it up
        2000-07 MC made it indep of Weibull and added parmin and parmax
        2018-08 MW ported to Python
    """
    # Validate input
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError('data must be a list or numpy array')

    if parmin is None:
        parmin = np.array([.005, 0., 0.])
    if parmax is None:
        parmax = np.array([.5, 10., .25])

    if data.shape[0] == 3:
        xx = data[0, :]
        nn = data[1, :]
        pp = data[2, :]
    else:
        raise ValueError('data must be m by 3 matrix')

    # here is where you effectively put the constraints.
    if (any(pars < parmin)) or (any(pars > parmax)):
        ll = 10000000
        return ll

    dispatcher = {
        'weibull': weibull,
        'weibull50': weibull50,
        'erf_psycho': erf_psycho,
        'erf_psycho_2gammas': erf_psycho_2gammas
    }
    try:
        probs = dispatcher[P_model](pars, xx)
    except KeyError:
        raise ValueError('invalid model, options are "weibull", ' +
                         '"weibull50", "erf_psycho" and "erf_psycho_2gammas"')

    assert (max(probs) <= 1) or (min(probs) >= 0), 'At least one of the probabilities is not ' \
                                                   'between 0 and 1'

    probs[probs == 0] = np.finfo(float).eps
    probs[probs == 1] = 1 - np.finfo(float).eps

    ll = - sum(nn * (pp * np.log(probs) + (1 - pp) * np.log(1 - probs)))
    return ll


def weibull(pars, xx):
    """
    Weibull function from 0 to 1, with lapse rate.
    Args:
        pars: Model parameters [alpha, beta, gamma].
        xx: vector of stim levels.
    Returns:
        A vector of length xx
    Raises:
        ValueError: pars must be of length 3
        TypeError: pars must be list-like or numpy array
    Information:
        1999-11 FH wrote it
        2000-01 MC cleaned it up
        2018-08 MW ported to Python
    """
    # Validate input
    if not isinstance(pars, (list, tuple, np.ndarray)):
        raise TypeError('pars must be list-like or numpy array')

    if len(pars) != 3:
        raise ValueError('pars must be of length 3')

    alpha, beta, gamma = pars
    return (1 - gamma) - (1 - 2 * gamma) * np.exp(-((xx / alpha) ** beta))


def weibull50(pars, xx):
    """
    Weibull function from 0.5 to 1, with lapse rate.
    Args:
        pars: Model parameters [alpha, beta, gamma].
        xx: vector of stim levels.
    Returns:
        A vector of length xx
    Raises:
        ValueError: pars must be of length 3
        TypeError: pars must be list-like or numpy array
    Information:
        2000-04 MC wrote it
        2018-08 MW ported to Python
    """
    # Validate input
    if not isinstance(pars, (list, tuple, np.ndarray)):
        raise TypeError('pars must be list-like or numpy array')

    if len(pars) != 3:
        raise ValueError('pars must be of length 3')

    alpha, beta, gamma = pars
    return (1 - gamma) - (.5 - gamma) * np.exp(-((xx / alpha) ** beta))


def erf_psycho(pars, xx):
    """
    erf function from 0 to 1, with lapse rate.
    Args:
        pars: Model parameters [bias, slope, lapse].
        xx: vector of stim levels.
    Returns:
        ff: A vector of length xx
    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> xx = np.arange(-50, 50)
        >>> ff = erf_psycho(np.array([-10., 10., 0.1]), xx)
        >>> plt.plot(xx, ff)
    Raises:
        ValueError: pars must be of length 3
        TypeError: pars must be a list or numpy array
    Information:
        2000    MC wrote it
        2018-08 MW ported to Python
    """
    # Validate input
    if not isinstance(pars, (list, tuple, np.ndarray)):
        raise TypeError('pars must be list-like or numpy array')

    if len(pars) != 3:
        raise ValueError('pars must be of length 4')

    (bias, slope, gamma) = pars
    return gamma + (1 - 2 * gamma) * (erf((xx - bias) / slope) + 1) / 2


def erf_psycho_2gammas(pars, xx):
    """
    erf function from 0 to 1, with two lapse rates.
    Args:
        pars: Model parameters [bias, slope, gamma].
        xx: vector of stim levels (%)
    Returns:
        ff: A vector of length xx
    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> xx = np.arange(-50, 50)
        >>> ff = erf_psycho_2gammas(np.array([-10., 10., 0.2, 0.]), xx)
        >>> plt.plot(xx, ff)
    Raises:
        ValueError: pars must be of length 4
        TypeError: pars must be list-like or numpy array
    Information:
        2000    MC wrote it
        2018-08 MW ported to Python
    """
    # Validate input
    if not isinstance(pars, (list, tuple, np.ndarray)):
        raise TypeError('pars must be a list-like or numpy array')

    if len(pars) != 4:
        raise ValueError('pars must be of length 4')

    (bias, slope, gamma1, gamma2) = pars
    return gamma1 + (1 - gamma1 - gamma2) * (erf((xx - bias) / slope) + 1) / 2
