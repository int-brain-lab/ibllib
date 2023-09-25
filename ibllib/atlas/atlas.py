"""
Classes for manipulating brain atlases, insertions, and coordinates.
"""

import warnings
import iblatlas.atlas


def deprecated_decorator(function):
    def deprecated_function(*args, **kwargs):
        warning_text = f"{function.__module__}.{function.__name__} is deprecated. " \
                       f"Use iblatlas.{function.__module__.split('.')[-1]}.{function.__name__} instead"
        warnings.warn(warning_text, DeprecationWarning)
        return function(*args, **kwargs)

    return deprecated_function


@deprecated_decorator
def BrainCoordinates(*args, **kwargs):
    return iblatlas.atlas.BrainCoordinates(*args, **kwargs)


@deprecated_decorator
def BrainAtlas(*args, **kwargs):
    return iblatlas.atlas.BrainAtlas(*args, **kwargs)


class Trajectory(iblatlas.atlas.Trajectory):
    """
    3D Trajectory (usually for a linear probe), minimally defined by a vector and a point.

    Examples
    --------
    Instantiate from a best fit from an n by 3 array containing xyz coordinates:

    >>> trj = Trajectory.fit(xyz)
    """


class Insertion(iblatlas.atlas.Insertion):
    """
    Defines an ephys probe insertion in 3D coordinate. IBL conventions.

    To instantiate, use the static methods: `Insertion.from_track` and `Insertion.from_dict`.
    """


@deprecated_decorator
def AllenAtlas(*args, **kwargs):
    return iblatlas.atlas.AllenAtlas(*args, **kwargs)


@deprecated_decorator
def NeedlesAtlas(*args, **kwargs):
    """
    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system. The Needles atlas defines a stretch along AP
    axis and a squeeze along the DV axis.

    Parameters
    ----------
    res_um : {10, 25, 50} int
        The Atlas resolution in micrometres; one of 10, 25 or 50um.
    **kwargs
        See AllenAtlas.

    Returns
    -------
    AllenAtlas
        An Allen atlas object with MRI atlas scaling applied.

    Notes
    -----
    The scaling was determined by manually transforming the DSURQE atlas [1]_ onto the Allen CCF.
    The DSURQE atlas is an MRI atlas acquired from 40 C57BL/6J mice post-mortem, with 40um
    isometric resolution.  The alignment was performed by Mayo Faulkner.
    The atlas data can be found `here <http://repo.mouseimaging.ca/repo/DSURQE_40micron_nifti/>`__.
    More information on the dataset and segmentation can be found
    `here <http://repo.mouseimaging.ca/repo/DSURQE_40micron/notes_on_DSURQE_atlas>`__.

    References
    ----------
    .. [1] Dorr AE, Lerch JP, Spring S, Kabani N, Henkelman RM (2008). High resolution
       three-dimensional brain atlas using an average magnetic resonance image of 40 adult C57Bl/6J
       mice. Neuroimage 42(1):60-9. [doi 10.1016/j.neuroimage.2008.03.037]
    """

    return iblatlas.atlas.NeedlesAtlas(*args, **kwargs)


@deprecated_decorator
def MRITorontoAtlas(*args, **kwargs):
    """
    The MRI Toronto brain atlas.

    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system. The MRI Toronto atlas defines a stretch along AP
    a squeeze along DV *and* a squeeze along ML. These are based on 12 p65 mice MRIs averaged [1]_.

    Parameters
    ----------
    res_um : {10, 25, 50} int
        The Atlas resolution in micrometres; one of 10, 25 or 50um.
    **kwargs
        See AllenAtlas.

    Returns
    -------
    AllenAtlas
        An Allen atlas object with MRI atlas scaling applied.

    References
    ----------
    .. [1] Qiu, LR, Fernandes, DJ, Szulc-Lerch, KU et al. (2018) Mouse MRI shows brain areas
       relatively larger in males emerge before those larger in females. Nat Commun 9, 2615.
       [doi 10.1038/s41467-018-04921-2]
    """
    return iblatlas.atlas.MRITorontoAtlas(*args, **kwargs)


@deprecated_decorator
def FranklinPaxinosAtlas(*args, **kwargs):
    return iblatlas.atlas.FranklinPaxinosAtlas(*args, **kwargs)
