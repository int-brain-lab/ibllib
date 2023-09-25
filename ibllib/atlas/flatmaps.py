"""Techniques to project the brain volume onto 2D images for visualisation purposes."""

from ibllib.atlas import deprecated_decorator
from iblatlas import flatmaps


@deprecated_decorator
def FlatMap(**kwargs):
    return flatmaps.FlatMap(**kwargs)


@deprecated_decorator
def circles(N=5, atlas=None, display='flat'):
    """
    :param N: number of circles
    :param atlas: brain atlas at 25 m
    :param display: "flat" or "pyramid"
    :return: 2D map of indices, ap_coordinate, ml_coordinate
    """

    return flatmaps.circles(N=N, atlas=atlas, display=display)


@deprecated_decorator
def swanson(filename="swanson2allen.npz"):
    """
    FIXME Document! Which publication to reference? Are these specifically for flat maps?
     Shouldn't this be made into an Atlas class with a mapping or scaling applied?

    Parameters
    ----------
    filename

    Returns
    -------

    """

    return flatmaps.swanson(filename=filename)


@deprecated_decorator
def swanson_json(filename="swansonpaths.json", remap=True):
    """
    Vectorized version of the swanson bitmap file. The vectorized version was generated from swanson() using matlab
    contour to find the paths for each region. The paths for each region were then simplified using the
    Ramer Douglas Peucker algorithm https://rdp.readthedocs.io/en/latest/

    Parameters
    ----------
    filename
    remap

    Returns
    -------

    """
    return flatmaps.swanson_json(filename=filename, remap=remap)
