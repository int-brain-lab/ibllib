import numpy as np
from numpy import linalg
import numba as nb
from typing import Tuple
import warnings
# from ibllib.pipes.mesoscope_tasks import surface_normal
# def plane_normal_form(face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """form a plane from a face (=3 points)

#     Args:
#         face (np.ndarray): a (3,3) array, dim1 = xyz

#     Returns:
#         Tuple[np.ndarray, np.ndarray]: a tuple of the plane in normal form. p0 = point on plane, n = normal
    
#     TODO Replace with direct call to surface_normal
#     """
#     return face[0], surface_normal(face)
#     p0, p1, p2 = face
#     n = np.cross(p0 - p1, p0 - p2)
#     n /= linalg.norm(n)
#     return p0, n


@nb.njit("Tuple((float64[:], float64[:]))(float64[:,:])")
def plane_normal_form(face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """form a plane from a face (=3 points)

    Args:
        face (np.ndarray): a (3,3) array, dim1 = xyz

    Returns:
        Tuple[np.ndarray, np.ndarray]: a tuple of the plane in normal form. p0 = point on plane, n = normal
    TODO Replace with direct call to surface_normal
    """

    p0, p1, p2 = face
    n = np.cross(p0 - p1, p0 - p2)
    n /= linalg.norm(n)
    return p0, n


# numba version fails with division by zero
def intersect_line_plane(
    ln0: np.ndarray, ln: np.ndarray, p0: np.ndarray, n: np.ndarray, warn=True
) -> np.ndarray:
    """return the intersection point of a line defined by l0 and l and plane in normal form p0 and n.

    derivation: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

    point on line is p = ln0 + d * ln
    point on plane is (p - p0).n = 0
    subsitute and solve for d
    ((ln0 + d * ln) - p0).n = 0

    Note:
    this function works in numpy, as if this fails ( = no intersection point) a warning is raised
    numba fails with ZeroDivisionError which can not be caught and handled

    Args:
        ln0 (np.ndarray): point on line
        ln (np.ndarray): line vector
        p0 (np.ndarray): point on plane
        n (np.ndarray): plane normal
        warn (bool, optional): warn or not. Defaults to True.

    Returns:
        np.ndarray: the intersection point
    """
    #

    if warn:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Catch all warnings
            d = np.dot(p0 - ln0, n) / np.dot(ln, n)
            if w:
                print(w, ln0, ln, p0, n)
            return ln0 + d.reshape(-1, 1) * ln
    else:
        d = np.dot(p0 - ln0, n) / np.dot(ln, n)
        return ln0 + d.reshape(-1, 1) * ln


@nb.njit("float64[:](float64[:],float64[:],float64[:],float64[:])")
def intersect_line_plane_nb(
    ln0: np.ndarray, ln: np.ndarray, p0: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """see intersect_line_plane() docstring"""
    # can only be called if we are sure such intersection point exists
    d = np.dot(p0 - ln0, n) / np.dot(ln, n)
    return ln0 + d * ln


@nb.njit("bool_(float64[:,:], float64[:])")
def point_in_face(face: np.ndarray, point: np.ndarray) -> np.bool_:
    """check if the point is within the triangular face

    3d form, baycentric coorinate based
    https://math.stackexchange.com/questions/2582202/does-a-3d-point-lie-on-a-triangular-plane

    Args:
        face (np.ndarray): a (3,3) array
        point (np.ndarray): a (3,1) array

    Returns:
        np.bool_: True if point in face
    """

    ph = np.append(point, 1)
    A = np.ones((4, 3))
    A[:-1, :] = face.T  # numba can't deal well with concatenate
    w = linalg.pinv(A.T @ A) @ A.T @ ph
    return np.all(np.logical_and(w > 0, w < 1))


def point_in_face_np(face: np.ndarray, point: np.ndarray) -> np.bool_:
    """see docstring of point_in_face() , numpy version"""
    ph = np.concatenate([point, np.ones(1)])[:, np.newaxis]
    A = np.concatenate([face.T, np.ones(3)[np.newaxis, :]], axis=0)
    w = linalg.pinv(A.T @ A) @ A.T @ ph
    return np.all(np.logical_and(w > 0, w < 1))


def intersect_line_mesh_np(
    vertices: np.ndarray,
    mesh_connectivity: np.ndarray,
    line_point: np.ndarray,
    line_vector: np.ndarray,
    numba: bool = False,
    exclude: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """calculates the intersection of a line with a mesh. Finds the intersected vertices,
    their indices, and the intersection points

    Args:
        vertices (np.ndarray): _description_
        mesh_connectivity (np.ndarray): _description_
        line_point (np.ndarray): _description_
        line_vector (np.ndarray): _description_
        numba (bool, optional): _description_. Defaults to False.
        exclude (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    # returns the index of the intersected faces
    ix = []
    faces = []
    intersection_points = []
    for i in range(mesh_connectivity.shape[0]):
        face = vertices[mesh_connectivity[i]]
        plane_point, plane_normal = plane_normal_form(face)
        if exclude and plane_normal[2] == 0:
            # this excludes triangles from the mesh that can not be intersected
            continue
        if numba:
            func = intersect_line_plane_nb
        else:
            func = intersect_line_plane

        intersection_point = func(line_point, line_vector, plane_point, plane_normal)
        if point_in_face(face, intersection_point):
            intersection_points.append(intersection_point)
            faces.append(face)
            ix.append(i)
    return (
        np.array(faces).astype("float64"),
        np.array(intersection_points).astype("float64"),
        np.array(ix).astype("uint64"),
    )


@nb.njit("float64(float64[:],float64[:])")
def get_angle(a, b):
    # the angle between two vectors a and b
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@nb.njit(
    "Tuple((float64[:,:,:], float64[:,:], int64[:]))(float64[:,:], int32[:,:], float64[:], float64[:])",
    parallel=True,
)
def intersect_line_mesh_nb(
    vertices: np.ndarray,
    mesh_connectivity: np.ndarray,
    line_point: np.ndarray,
    line_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """see intersect_line_mesh_np"""
    N = mesh_connectivity.shape[0]
    vertices_to_check = np.zeros(N, dtype="bool")

    for i in nb.prange(N):
        face = vertices[mesh_connectivity[i]]
        plane_point, plane_normal = plane_normal_form(face)
        # test if plane normal and line vector are parallel
        # if not, they will have an intersection point
        tol = 1e-5
        alpha = get_angle(plane_normal, line_vector)
        # exclude faces that will not be intersected
        if np.abs((np.abs(alpha) - np.pi / 2)) > tol:
            face = vertices[mesh_connectivity[i]]
            plane_point, plane_normal = plane_normal_form(face)
            intersection_point = intersect_line_plane_nb(
                line_point, line_vector, plane_point, plane_normal
            )
            if point_in_face(face, intersection_point):
                vertices_to_check[i] = True

    ix = np.where(vertices_to_check)[0]
    intersection_points = np.zeros((ix.shape[0], 3), dtype="float64")
    faces = np.zeros((ix.shape[0], 3, 3), dtype="float64")
    for j in nb.prange(ix.shape[0]):
        i = ix[j]
        face = vertices[mesh_connectivity[i]]
        plane_point, plane_normal = plane_normal_form(face)
        intersection_points[j] = intersect_line_plane_nb(
            line_point, line_vector, plane_point, plane_normal
        )
        faces[j] = face

    return (
        faces,
        intersection_points,
        ix,
    )


@nb.njit("Tuple((float64[:,:], int64))(float64[:,:,:],float64[:])")
def get_closest_face(
    faces: np.ndarray, point: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """find the face closest to the point. For this, face coordinates are averaged.

    Args:
        faces (np.ndarray): array of shape (N, 3, 3) for N faces
        point (np.ndarray): array of shape (3,)

    Returns:
        Tuple[np.ndarray, np.ndarray]: the closest face and it's index
    """
    # faces is
    dists = np.array(
        [linalg.norm(point - np.average(face, axis=0)) for face in faces],
        dtype="float64",
    )
    min_ix = np.argmin(dists)
    return faces[min_ix], min_ix


# numpy version
def find_closest_point_from_line_np(
    points: np.ndarray, l0: np.ndarray, l: np.ndarray
) -> np.ndarray:
    """numpy variant of find_closest_point_from_line_nb"""
    ds = linalg.norm((l0 - points) - np.dot(l0 - points, l)[:, np.newaxis] * l, axis=1)
    point = points[np.argmin(ds)]
    return point


# numba compatible version
@nb.njit("float64[:](float64[:,:], float64[:], float64[:])")
def find_closest_point_from_line_nb(
    points: np.ndarray, l0: np.ndarray, l: np.ndarray
) -> np.ndarray:
    """for a given set of points, return the point that is closest to the line
    (defined by a point and a vector)

    Args:
        points (np.ndarray): the points to evaluate
        l0 (np.ndarray): point on the line
        l (np.ndarray): vector of the line

    Returns:
        np.ndarray: the closest point
    """
    vs = (l0 - points) - np.dot(l0 - points, l)[:, np.newaxis] * l
    ds = np.array([linalg.norm(v) for v in vs])
    point = points[np.argmin(ds)]
    return point


@nb.njit("float64[:,:](float64[:,:],float64[:,:],float64[:])", parallel=True)
def find_closest_points_on_surface(
    points_eval: np.ndarray, brain_surface_points: np.ndarray, n: np.ndarray
) -> np.ndarray:
    # TODO this needs heavy refactoring
    # rename into: find_closest_point_from_lines
    # as this is essentially a parallelizatoin wrapper find_closest_point_from_line_nb
    # change the call signature accordingly
    N = points_eval.shape[0]
    points_closest = np.zeros((N, 3))
    for i in nb.prange(N):
        points_closest[i, :] = find_closest_point_from_line_nb(
            brain_surface_points, points_eval[i], n
        )
    return points_closest


def get_rotation_between_vectors(a: np.ndarray, b: np.ndarray, as_affine=True):
    # returns the (3,3) transform or (4,4)
    # https://math.stackexchange.com/a/2470436

    # extend this to constrain one axis
    i = a
    ip = b
    j = np.cross(i, ip) / linalg.norm(np.cross(i, ip))
    jp = j
    k = np.cross(i, j)
    kp = np.cross(ip, jp)

    R = np.stack([ip, jp, kp], axis=1) @ np.stack([i, j, k], axis=1).T
    if as_affine:
        R_ = np.zeros((4, 4))
        R_[-1, -1] = 1
        R_[:3, :3] = R
        return R_
    else:
        return R


def get_vector_angles(v: np.ndarray, in_radians=True):
    # follows IBL conventions?
    # TODO verify the angles, again ...

    ml, ap, dv = v
    # theta is the angle for rotation around the ml axis = in plane in (ap, dv)
    # == pitch

    a = np.array([ap, dv])
    b = np.array([0, 1])
    theta = np.arccos(np.dot(a, b) / (linalg.norm(a) * linalg.norm(b)))

    # phi is the angle for rotation around the dv axis = in plane in (ml, ap)
    # == yaw
    a = np.array([ml, ap])
    b = np.array([1, 0])
    phi = np.arccos(np.dot(a, b) / (linalg.norm(a) * linalg.norm(b)))
    # beta is the angle for rotation in AP axis = in plane in (ml, dv)
    # == roll
    # not clearly defined in the IBL image (extent)

    a = np.array([ml, dv])
    b = np.array([0, 1])
    beta = np.arccos(np.dot(a, b) / (linalg.norm(a) * linalg.norm(b)))

    angles = np.array([phi, theta, beta])
    if in_radians:
        return angles
    else:
        return angles * 360 / (2 * np.pi)
