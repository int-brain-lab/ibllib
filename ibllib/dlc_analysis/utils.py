import numba
from numba import jit
import statsmodels.api as sm
from scipy.interpolate import (
    InterpolatedUnivariateSpline,
    UnivariateSpline,
    CubicSpline,
)

#
import numpy as np
from itertools import combinations
import time


def mask_out_box(xrm, yrm, y_box=[400, 950], x_box=[300, 1281]):
    """
    Return copies of xrm and yrm arrays where elements outside
    of box determined by x_box and y_box are set to zero.
    y_box = [ymin, ymax] , x_box = [xmin, xmax]
    mask  = (y < ymin or y > ymax) or  (x < xmin or x > xmax)
    Inputs:
    ______
    :param xrm: np.array
        values to be compared with x_box
    :param yrm: np.array
        values to be compared with y_box
    :param y_box: tuple as list
        tuple of min and max values that elements in yrm can take
    :param x_box: tuple as list
        tuple of min and max values that elements in xrm can take
    :return:
    xr : np.array
        copy of xrm array where elements
    yr : np.array,
    """
    # Create mask given box
    xr = xrm.copy()
    yr = yrm.copy()
    mask_out = np.ones(xr.shape).astype("bool")
    mask_out[yr < y_box[0]] = 0
    mask_out[yr > y_box[1]] = 0
    mask_out[xr < x_box[0]] = 0
    mask_out[xr > x_box[1]] = 0

    # mask out bad frames
    xr[~mask_out] = np.nan
    yr[~mask_out] = np.nan

    return xr, yr


def find_slices_group(maskr, idx_group=None):
    """
    Given an array (maskr) returns the list of indices of
    consecutive segments (slices) and their lengths (slices_lens)
    which do not contain np.nan elements.
    The variable idx_group sets the rows of maskr to consider.
    This functions considers all the rows by default.
    Inputs:
    ______
    :param maskr:  np.array (D, T)
        array where we look for continuous segments
    :param idx_group: list
        list of indices to consider to look for continuous segments
    :return:
    slices: list of "slice" objects
        each "slice" object contains the [start, stop] indices of a
         consecutive segment without np.nans.
    slices_lens: np.array
        each element in array is the length of each consecutive segment.
    """
    # set default
    D, _ = maskr.shape

    if idx_group is None:
        idx_group = np.arange(D)
    else:
        assert np.max(idx_group) <= D

    # Identify sequence of elements
    a = maskr[np.ix_(idx_group)].sum(0)
    # mask array where the sequences are not complete
    b = np.ma.masked_less(a, len(idx_group))
    # list of slices corresponding to the unmasked groups
    slices = np.ma.clump_unmasked(b)
    # lengths of each slice
    slices_lens = np.asarray([slc.stop - slc.start for slc in slices])
    print("group {}: \t max {}".format(idx_group, slices_lens.max()))
    return slices, slices_lens


def find_slices_group_lens(maskr, idx_group=None, len_combinations=None):
    """
    Creates a dictionary slices_groups[group] = array
    where each "group" key is an element of idx_groups or their  combinations
    (in combos of len_combinations). The value of each key is an array
    of lengths of continuous segments if we only consider the "group" rows in maskr.
    Inputs:
    ______
    :param maskr: np.array (D, T)
        array where we look for continuous segments
    :param idx_group: list
        list of indices to consider to look for continuous segments
    :param len_combinations: list
        list of size of combinations of elements in idx_group
    :return:
    slices_groups: dictionary
        where each key is a list of rows in maskr to find continuous segments,
        and the respective value is an array of the lengths of continuous segments.

    """
    D, _ = maskr.shape

    # array indices
    if idx_group is None:
        idx_group = np.arange(D)

    # combinations of array indices
    if len_combinations is None:
        len_combinations = np.arange(1, D + 1)

    # For different groups
    slices_groups = {}
    for num_fingers in len_combinations:
        for idx_ in combinations(idx_group, num_fingers):
            # find slices and slices' lengths
            slides_, slices_lens = find_slices_group(maskr, idx_)
            # assign groups to each group
            slices_groups[str(idx_)] = slices_lens
    return slices_groups


def find_top_continuous_slice(xr, min_len=0, burn_in=0, chunks=True):
    """
    Find list of arrays, where each array is a subset of xr
    without np.nan elements.
    The length of the array must be greater than min_len,
    If burn_in is greater than 0, we discard any segments of the data
    greater than that segment.
    Inputs:
    ______
    :param xr: np.array (D, T)
        array where to look for continuous segments
    :param min_len: int
        minimum length of any array
    :param burn_in: int
        number of samples to discard at the beginning of each continous segment
    :return:
    datas: list of arrays
        each array is a continuous segments in xr
    datas_slices: list of slices
        each slice object is the (start and stop points) of each array in datas
    datas_chunks: list of arrays
        each array contains the indices of each dataset in the original xr array.
    """
    ndim = np.ndim(xr)

    if ndim == 1:
        xr = xr[None, :]

    maskm = ~np.isnan(xr)
    slices, slices_lens = find_slices_group(maskm)

    # Sorted according to length
    idx_sorted = np.argsort(slices_lens)[::-1]

    # Create list for outputs
    datas = []
    datas_slices = []

    if chunks:
        datas_chunks = []

    #
    for idx_slice in idx_sorted:
        if slices_lens[idx_slice] < min_len + burn_in:
            break

        mslice = slices[idx_slice]
        tx = xr[:, mslice]

        mchunk = np.arange(mslice.start, mslice.stop)

        if burn_in > 0:
            mslice = slice(mslice.start + burn_in, mslice.stop - burn_in)
            tx = tx[:, burn_in:-burn_in]

        if ndim == 1:
            tx = tx.flatten()
        datas.append(tx.T)
        datas_slices.append(mslice)
        if chunks:
            datas_chunks.append(mchunk)

    print("return {} slices".format(len(datas)))
    if chunks:
        return datas, datas_slices, datas_chunks
    else:
        return datas, datas_slices


def get_trace_fixed_count(x, chunk=2000, number_nans=50):
    """
    Find the start and end index of of a chunk of x where
    there are at least "number_nans" elements as np.nan
    :param x: np.array (D, )
    :param chunk: int
    :param number_nans: int
    :return:
    tstart: int
        start of segment
    tend: int
        end of segment
    """
    counts = count_trace_chunks(x, chunk=chunk)
    # Select a trace where the loss is closest to 50
    T = len(x)
    tstart = np.argmin(np.abs(counts - number_nans)) * chunk
    tend = min(tstart + chunk, T)
    return tstart, tend


@jit(nopython=True)
def count_trace_chunks(x, chunk=2000):
    """
    Return array of sum of elements in x
    split evenly in chunks of length chunk.
    :param x: np.array (T, )
    :param chunk: int
        length of each segment of x to consider
    :return:
    counts: np.array (T // chunk+1, )
        each element is the sum of a chunk of x
    """
    T = len(x)
    counts = np.zeros(T // chunk + 1)
    for ii, start in enumerate(range(0, T, chunk)):
        stop = min(start + chunk, T)
        counts[ii] = np.sum(x[start:stop])
    return counts


def smooth_trace(y, box_pts=3):
    """
    Smooth with a box of length box_pts
    :param y: np.array (T,)
    :param box_pts: int
        length of box used to smooth trace
    :return:
    y_smooth: np.array(T, )
        smoothed trace
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def convertparms2start(pn):
    """
    Creating a start value for sarimax in case of an value error
    See: https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk
    Edited from Github/AlexEMG/DeepLabCut
    """
    if "ar." in pn:
        return 0
    elif "ma." in pn:
        return 0
    elif "sigma" in pn:
        return 1
    else:
        return 0


def FitSARIMAXModel(
    x,
    p=None,
    pcutoff=0.01,
    alpha=0.001,
    ARdegree=3,
    MAdegree=2,
    nforecast=0,
    min_num_bad_frames=10,
):
    """
    Fits a Seasonal Autoregressive Integrated Moving-Average with
    eXogenous regressors (SARIMAX) to x
    Edited from Github/AlexEMG/DeepLabCut
    :param x: np.array (T,)
        input array
    :param p: np.array (T,)
        level of confidence elements of x
    :param pcutoff: float
        maximum level of confidence in elements of x,
        any element with confidence lower than pcutoff will be discarded
    :param alpha: float
        value for confidence interval
    :param ARdegree: int
        autoregressive degree of SARIMAX model
    :param MAdegree: int
        moving average degree of SARIMAX model
    :param nforecast: int
        number of elements for lookhead in SARIMAX model
    :param min_num_bad_frames: int
        minimum number of bad elements in x.
        If x has is missing less than min_num_bad_frames, this function
        returns the same value
    :return:
    predicted_mean: np,array (T,)
        value of predicted x
        if algorithm fails, an np.zeros array is returned
    conf_int: np,array(2, T)
        confidence intervals for prediction in predicted_mean
        if algorithm fails, an np.zeros array is returned

    """
    start = time.time()
    if p is None:
        p = np.ones(x.shape)

    # x is the raw signal
    # p is the likelihood
    # p cutoff is our uncertainty

    # see
    # http://www.statsmodels.org/stable/statespace.html#seasonal-autoregressive-integrated-moving-average-with-exogenous-regressors-sarimax
    Y = x.copy()
    Y[p < pcutoff] = np.nan  # Set uncertain estimates to nan (modeled as missing data)

    # is there are nans in at least min_frames # of frames
    if np.sum(np.isfinite(Y)) > min_num_bad_frames:
        print(
            "There are at least {} bad frames. Running SARIMAX model".format(
                min_num_bad_frames
            )
        )
        # SARIMAX implementation has better prediction models than simple ARIMAX
        # (however we do not use the seasonal etc. parameters!)
        mod = sm.tsa.statespace.SARIMAX(
            Y.flatten(),
            order=(ARdegree, 0, MAdegree),
            seasonal_order=(0, 0, 0, 0),
            simple_differencing=True,
        )

        # Autoregressive Moving Average ARMA(p,q) Model
        # mod = sm.tsa.ARIMA(Y, order=(ARdegree,0,MAdegree)) #order=(ARdegree,0,MAdegree)
        try:
            res = mod.fit(disp=False)
        except ValueError:
            # https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk (let's update to statsmodels 0.10.0 soon...)
            startvalues = np.array([convertparms2start(pn) for pn in mod.param_names])
            res = mod.fit(start_params=startvalues, disp=False)

        predict = res.get_prediction(end=mod.nobs + nforecast)
        print(time.time() - start)

        return predict.predicted_mean[1:], predict.conf_int(alpha=alpha)[1:]
    else:
        return np.nan * np.zeros(len(Y)), np.nan * np.zeros((len(Y), 2))


#%%

def interpolate_multiple_traces(xr):
    """
    Interpolate each row in array xr
    See function interpolate_trace for additional details
    :param xr: np.array (D, T)
        array to be interpolated
    :return:
    xr_hat: np.array (D, T)
        array with np.nan elements are interpolated
    """
    xr_hat = np.zeros(xr.shape)

    for ii, x in enumerate(xr):
        x_hat = interpolate_trace(x)
        xr_hat[ii, :] = x_hat

    return xr_hat

def interpolate_trace(x, kind="cubic", axis=-1, interp_type="cubicspline"):
    """
    Interpolate array x using scipy interpolators
    :param x: np.array (D, )
    :param kind: string
        degree of polynomial interpolator to use if interp_type='poly'
    :param axis: int
        axis along which to interpolate trace if interp_type='poly'
    :param interp_type: string : ['poly', 'unispline','spline', cubicspline]
    :return:
    trace_x_hat: np.array(D, )
        interpolated trace
    """
    from scipy.interpolate import interp1d

    # assume x is T x D
    T = len(x)
    # time_frames ratios

    time_frames = np.arange(T)

    # bad frames
    if np.ndim(x) == 1:
        mask_frames = np.isnan(x)
    else:
        # assume mask shared across dimensions
        mask_frames = np.isnan(x[:, 0])
    # mask_frames = np.where(~mask_trace)[0]
    # from scipy.interpolate import interp1d`

    if interp_type is "poly":
        f = interp1d(
            time_frames[~mask_frames],
            x[~mask_frames],
            kind=kind,
            fill_value="extrapolate",
            axis=axis,
        )
    elif interp_type is "unispline":
        f = InterpolatedUnivariateSpline(time_frames[~mask_frames], x[~mask_frames])
    elif interp_type is "spline":
        f = UnivariateSpline(time_frames[~mask_frames], x[~mask_frames])

    elif interp_type is "cubicspline":
        f = CubicSpline(time_frames[~mask_frames], x[~mask_frames])

    trace_x_hat = f(time_frames)

    return trace_x_hat


def z_score(x, axis=0, th=2):
    """
    Return z score values excluding nans
    :param x: np.array
    :param axis: int
        axis along which to calculate mean and std of x
    :param th: float
        threshold to calculate z-score
    :return:
    metric : np.array
        array of boolean values where x >= mean + th*std
    """
    # mask = np.isnan(x)
    metric = np.abs(x) >= np.nanmean(x, axis=axis, keepdims=True) + th * np.nanstd(
        x, axis=axis, keepdims=True
    )
    # metric[mask] = True
    return metric

#%%

