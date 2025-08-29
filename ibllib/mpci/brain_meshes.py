import numpy as np
from scipy.spatial import ConvexHull
from ibllib.mpci.linalg import (
    intersect_line_mesh,
    get_closest_face,
    plane_normal_form,
)
from iblatlas.atlas import BrainAtlas
from typing import Tuple


def load_surface_triangulation():
    # DEPRECATED
    path = "/home/georg/code/ibllib/ibllib/io/extractors/mesoscope/surface_triangulation.npz"
    surface_triangulation = np.load(path)
    points = surface_triangulation["points"].astype("f8")
    connectivity_list = surface_triangulation["connectivity_list"]
    surface_triangulation.close()
    return points, connectivity_list


def calculate_surface_triangulation(atlas: BrainAtlas) -> Tuple[np.ndarray, np.ndarray]:
    """runs surface triangulation on the brain atlas and returns the mesh two arrays: points and connectivity_list

    Args:
        atlas (BrainAtlas): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    points = get_surface_points(atlas, dropna=True)
    hull = ConvexHull(points)
    connectivity_list = hull.simplices
    return points, connectivity_list


def get_surface_points(atlas: BrainAtlas, dropna=True) -> np.ndarray:
    """for a given atlas, return all points that are on the brain surface in um.

    Args:
        atlas (BrainAtlas): _description_
        dropna (bool, optional): _description_. Defaults to True.

    Returns:
        np.ndarray: the surface points with shape (N,3) in (ml,ap,dv)
    """

    ap_grid, ml_grid = np.meshgrid(
        atlas.bc.yscale, atlas.bc.xscale
    )  # now this indexes into AP, ML
    points = (
        np.stack(
            [ml_grid.T.flatten(), ap_grid.T.flatten(), atlas.top.flatten()], axis=1
        )
        * 1e6  # <- converts the atlas into um
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
    """for a given ml,ap coordinates, returns the plane on the brain surface
    in normal form

    Args:
        ml (np.float64): _description_
        ap (np.float64): _description_
        vertices (np.ndarray): _description_
        connectivity_list (np.ndarray): _description_
        upwards (bool, optional): enforce the normal pointing upwards. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: plane as defined by point and normal
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
