import numpy as np
from scipy.spatial import ConvexHull
from ibllib.mpci.linalg import (
    intersect_line_mesh,
    get_closest_face,
    plane_normal_form,
)
from iblatlas.atlas import BrainAtlas
from typing import Tuple


def calculate_surface_triangulation(atlas: BrainAtlas) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run surface triangulation on the given brain atlas and return the mesh vertices and connectivity list.

    Parameters
    ----------
    atlas : BrainAtlas
        Brain atlas object containing anatomical data.

    Returns
    -------
    points : np.ndarray
        Array of 3D coordinates representing the surface points of the brain atlas.
    connectivity_list : np.ndarray
        Array of indices representing the connectivity (triangles) between surface points.

    Notes
    -----
    This function uses a convex hull to compute the triangulation of the surface points.
    """
    points = get_surface_points(atlas, dropna=True)
    hull = ConvexHull(points)
    connectivity_list = hull.simplices
    return points, connectivity_list


def get_surface_points(atlas: BrainAtlas, dropna=True) -> np.ndarray:
    """
    Returns all points on the brain surface in micrometers.

    Parameters
    ----------
    atlas : BrainAtlas
        The brain atlas object.
    dropna : bool, optional
        If True, drop points with NaN values. Default is True.

    Returns
    -------
    np.ndarray
        Surface points with shape (N, 3) in (ml, ap, dv) coordinates.
    """

    ap_grid, ml_grid = np.meshgrid(
        atlas.bc.yscale, atlas.bc.xscale
    )  # now this indexes into AP, ML
    points = (
        np.stack(
            [ml_grid.T.flatten(), ap_grid.T.flatten(), atlas.top.flatten()], axis=1
        )
        * 1e6  # <- converts the atlas into μm
    )
    if dropna:
        points = points[~np.isnan(points[:, 2])]
    return points


def get_plane_at_point_mlap(
    ml: np.float64,
    ap: np.float64,
    vertices: np.ndarray,
    connectivity_list: np.ndarray,
    upwards=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """For a given ml, ap coordinate, return the plane on the brain surface in normal form.

    Parameters
    ----------
    ml : np.float64
        The mediolateral coordinate in micrometers.
    ap : np.float64
        The anteroposterior coordinate in micrometers.
    vertices : np.ndarray
        The mesh vertices as an Nx3 array in (ml, ap, dv) coordinates.
    connectivity_list : np.ndarray
        The mesh connectivity list.
    upwards : bool
        Enforce the normal pointing upwards. Defaults to True.

    Returns
    -------
    np.ndarray
        One of the points on the surface plane.
    np.ndarray
        The normal vector of the surface plane.
    """
    # projects from a point above the brain downwards until it intersects
    # the mesh
    ln0 = np.array([ml, ap, 1000.0])
    ln = np.array([0.0, 0.0, -1.0])
    faces, ips, _ = intersect_line_mesh(vertices, connectivity_list, ln0, ln)
    face, ix = get_closest_face(faces, ln0)
    _, n = plane_normal_form(face)  # the brain normal
    p = ips[ix]  # the intersection point in the mesh triangle
    if upwards:
        if n[2] < 0:
            n *= -1
    return p, n
