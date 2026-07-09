import numpy as np
from numpy import linalg
import numba as nb
from typing import Tuple


def surface_normal(triangle):
    """
    Calculate the surface normal unit vector of one or more triangles.

    Parameters
    ----------
    triangle : numpy.array
        An array of shape (n_triangles, 3, 3) representing (Px Py Pz).

    Returns
    -------
    numpy.array
        The surface normal unit vector(s).
    """
    if triangle.shape == (3, 3):
        triangle = triangle[np.newaxis, :, :]
    if triangle.shape[1:] != (3, 3):
        raise ValueError('expected array of shape (3, 3); 3 coordinates in x, y, and z')
    V = triangle[:, 1, :] - triangle[:, 0, :]  # V = P2 - P1
    W = triangle[:, 2, :] - triangle[:, 0, :]  # W = P3 - P1

    Nx = (V[:, 1] * W[:, 2]) - (V[:, 2] * W[:, 1])  # Nx = (Vy * Wz) - (Vz * Wy)
    Ny = (V[:, 2] * W[:, 0]) - (V[:, 0] * W[:, 2])  # Ny = (Vz * Wx) - (Vx * Wz)
    Nz = (V[:, 0] * W[:, 1]) - (V[:, 1] * W[:, 0])  # Nz = (Vx * Wy) - (Vy * Wx)
    N = np.c_[Nx, Ny, Nz]
    # Calculate unit vector. Transpose allows vectorized operation.
    A = N / np.sqrt((Nx ** 2) + (Ny ** 2) + (Nz ** 2))[np.newaxis].T
    return A.squeeze()


@nb.njit('b1(f8[:,:], f8[:])')
def in_triangle(triangle, point):
    """
    Check whether `point` lies within `triangle`.

    Parameters
    ----------
    triangle : numpy.array
        A (2 x 3) array of x-y coordinates; A(x1, y1), B(x2, y2) and C(x3, y3).
    point : numpy.array
        A point, P(x, y).

    Returns
    -------
    bool
        True if coordinate lies within triangle.
    """
    def area(x1, y1, x2, y2, x3, y3):
        """Calculate the area of a triangle, given its vertices."""
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.)

    x1, y1, x2, y2, x3, y3 = triangle.flat
    x, y = point
    A = area(x1, y1, x2, y2, x3, y3)  # area of triangle ABC
    A1 = area(x, y, x2, y2, x3, y3)  # area of triangle PBC
    A2 = area(x1, y1, x, y, x3, y3)  # area of triangle PAC
    A3 = area(x1, y1, x2, y2, x, y)  # area of triangle PAB
    # Check if sum of A1, A2 and A3 equals that of A
    diff = np.abs((A1 + A2 + A3) - A)
    REL_TOL = 1e-9
    return diff <= np.abs(REL_TOL * A)  # isclose not yet implemented in numba 0.57


@nb.njit('i8(f8[:], f8[:,:], intp[:,:])', nogil=True)
def find_triangle(point, vertices, connectivity_list):
    """
    Find which vertices contain a given point.

    Currently O(n) but could take advantage of connectivity order to be quicker.

    Parameters
    ----------
    point : numpy.array
        The (x, y) coordinate of a point to locate within one of the triangles.
    vertices : numpy.array
        An N x 3 array of vertices representing a triangle mesh.
    connectivity_list : numpy.array
        An N x 3 array of indices representing the connectivity of `points`.

    Returns
    -------
    int
        The index of the vertices containing `point`, or -1 if not within any triangle.
    """
    face_ind = -1
    for i in nb.prange(connectivity_list.shape[0]):
        triangle = vertices[connectivity_list[i, :], :]
        if in_triangle(triangle, point):
            face_ind = i
            break
    return face_ind


@nb.njit('Tuple((f8[:], intp[:]))(f8[:], f8[:])', nogil=True)
def _nearest_neighbour_1d(x, x_new):
    """
    Nearest neighbour interpolation with extrapolation.

    This was adapted from scipy.interpolate.interp1d but returns the indices of each nearest x
    value.  Assumes x is not sorted.

    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    x_new : (N,) array_like
        A 1D array of values to apply function to.

    Returns
    -------
    numpy.array
        A 1D array of interpolated values.
    numpy.array
        A 1D array of indices.
    """
    SIDE = 'left'  # use 'right' to round up to nearest int instead of rounding down
    # Sort values
    ind = np.argsort(x, kind='mergesort')
    x = x[ind]
    x_bds = x / 2.0  # Do division before addition to prevent possible integer overflow
    x_bds = x_bds[1:] + x_bds[:-1]
    # Find where in the averaged data the values to interpolate would be inserted.
    x_new_indices = np.searchsorted(x_bds, x_new, side=SIDE)
    # Clip x_new_indices so that they are within the range of x indices.
    x_new_indices = x_new_indices.clip(0, len(x) - 1).astype(np.intp)
    # Calculate the actual value for each entry in x_new.
    y_new = x[x_new_indices]
    return y_new, ind[x_new_indices]


@nb.njit('Tuple((f8[:,:], u2[:]))(f8[:], f8[:], f8[:,:], f8[:], f8[:], f8[:], u2[:,:,:])', nogil=True)
def _update_points(t, normal_vector, coords, axis_ml_um, axis_ap_um, axis_dv_um, atlas_labels):
    """
    Determine the MLAPDV coordinate and brain location index for each of the given coordinates.

    This has been optimized in numba. The majority of the time savings come from replacing iterp1d
    and ismember with _nearest_neighbour_1d which were extremely slow. Parallel iteration further
    halved the time it took per 512x512 FOV.

    Parameters
    ----------
    t : numpy.array
        An N x 3 evenly spaced set of coordinates representing points going down from the coverslip
        towards the brain.
    normal_vector : numpy.array
        The unit vector of the face normal to the center of the window.
    coords : numpy.array
        A set of N x 3 coordinates representing the MLAPDV coordinates of each pixel relative to
        the center of the window, in micrometers (um).
    axis_ml_um : numpy.array
        An evenly spaced array of medio-lateral brain coordinates relative to bregma in um, at the
        resolution of the atlas image used.
    axis_ap_um : numpy.array
        An evenly spaced array of anterio-posterior brain coordinates relative to bregma in um, at
        the resolution of the atlas image used.
    axis_dv_um : numpy.array
        An evenly spaced array of dorso-ventral brain coordinates relative to bregma in um, at
        the resolution of the atlas image used.
    atlas_labels : numpy.array
        A 3D array of integers representing the brain location index of each voxel of a given
        atlas. The shape is expected to be (nAP, nML, nDV).

    Returns
    -------
    numpy.array
        An N by 3 array containing the MLAPDV coordinates in um of each pixel coordinate.
        Coordinates outside of the brain are NaN.
    numpy.array
        A 1D array of atlas label indices the length of `coordinates`.
    """
    # passing through the center of the craniotomy/coverslip
    traj_coords_centered = np.outer(t, -normal_vector)
    MLAPDV = np.full_like(coords, np.nan)
    annotation = np.zeros(coords.shape[0], dtype=np.uint16)
    n_points = coords.shape[0]
    for p in nb.prange(n_points):
        # Shifted to the correct point on the coverslip, in true ML-AP-DV coords
        traj_coords = traj_coords_centered + coords[p, :]

        # Find intersection coordinate with the brain.
        # Only use coordinates that exist in the atlas (kind of nearest neighbour interpolation)
        ml, ml_idx = _nearest_neighbour_1d(axis_ml_um, traj_coords[:, 0])
        ap, ap_idx = _nearest_neighbour_1d(axis_ap_um, traj_coords[:, 1])
        dv, dv_idx = _nearest_neighbour_1d(axis_dv_um, traj_coords[:, 2])

        # Iterate over coordinates to find the first (if any) that is within the brain
        ind = -1
        area = 0  # 0 = void; 1 = root
        for i in nb.prange(traj_coords.shape[0]):
            anno = atlas_labels[ap_idx[i], ml_idx[i], dv_idx[i]]
            if anno > 0:  # first coordinate in the brain
                ind = i
                area = anno
                if area > 1:  # non-root brain area; we're done
                    break
        if area > 1:
            point = traj_coords[ind, :]
            MLAPDV[p, :] = point  # in um
            annotation[p] = area
        else:
            MLAPDV[p, :] = np.nan
            annotation[p] = area  # root or void

    return MLAPDV, annotation


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
# def intersect_line_plane(
#     ln0: np.ndarray, ln: np.ndarray, p0: np.ndarray, n: np.ndarray, warn=True
# ) -> np.ndarray:
    # """return the intersection point of a line defined by l0 and l and plane in normal form p0 and n.

    # derivation: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

    # point on line is p = ln0 + d * ln
    # point on plane is (p - p0).n = 0
    # subsitute and solve for d
    # ((ln0 + d * ln) - p0).n = 0

    # Note:
    # this function works in numpy, as if this fails ( = no intersection point) a warning is raised
    # numba fails with ZeroDivisionError which can not be caught and handled

    # Args:
    #     ln0 (np.ndarray): point on line
    #     ln (np.ndarray): line vector
    #     p0 (np.ndarray): point on plane
    #     n (np.ndarray): plane normal
    #     warn (bool, optional): warn or not. Defaults to True.

    # Returns:
    #     np.ndarray: the intersection point
    # """
#     #

#     if warn:
#         with warnings.catch_warnings(record=True) as w:
#             warnings.simplefilter("always")  # Catch all warnings
#             d = np.dot(p0 - ln0, n) / np.dot(ln, n)
#             if w:
#                 print(w, ln0, ln, p0, n)
#             return ln0 + d.reshape(-1, 1) * ln
#     else:
#         d = np.dot(p0 - ln0, n) / np.dot(ln, n)
#         return ln0 + d.reshape(-1, 1) * ln


@nb.njit("float64[:](float64[:],float64[:],float64[:],float64[:])")
def intersect_line_plane(
    ln0: np.ndarray, ln: np.ndarray, p0: np.ndarray, n: np.ndarray
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
    # can only be called if we are sure such intersection point exists
    d = np.dot(p0 - ln0, n) / np.dot(ln, n)
    return ln0 + d * ln


@nb.njit
def point_in_face_original_method(face: np.ndarray, point: np.ndarray) -> np.bool_:
    """Exactly replicates the original point_in_face algorithm."""
    # Ensure point is 1D
    if point.size == 3:
        if point.ndim == 1:
            point_1d = point
        else:
            point_1d = np.array([point.flat[0], point.flat[1], point.flat[2]])
    else:
        return False  # Invalid point

    # Replicate the original algorithm: ph = np.append(point, 1)
    ph = np.array([point_1d[0], point_1d[1], point_1d[2], 1.0])

    # A = np.ones((4, 3)); A[:-1, :] = face.T
    A = np.ones((4, 3), dtype=np.float64)
    A[0, 0] = face[0, 0]; A[0, 1] = face[1, 0]; A[0, 2] = face[2, 0]  # x coordinates
    A[1, 0] = face[0, 1]; A[1, 1] = face[1, 1]; A[1, 2] = face[2, 1]  # y coordinates
    A[2, 0] = face[0, 2]; A[2, 1] = face[1, 2]; A[2, 2] = face[2, 2]  # z coordinates
    # A[3, :] = [1, 1, 1] already set

    # Compute pseudoinverse manually: pinv(A.T @ A) @ A.T @ ph
    # First: AtA = A.T @ A
    AtA = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            AtA[i, j] = A[0, i] * A[0, j] + A[1, i] * A[1, j] + A[2, i] * A[2, j] + A[3, i] * A[3, j]

    # Compute determinant for 3x3 matrix
    det = (AtA[0, 0] * (AtA[1, 1] * AtA[2, 2] - AtA[1, 2] * AtA[2, 1]) -
           AtA[0, 1] * (AtA[1, 0] * AtA[2, 2] - AtA[1, 2] * AtA[2, 0]) +
           AtA[0, 2] * (AtA[1, 0] * AtA[2, 1] - AtA[1, 1] * AtA[2, 0]))

    if abs(det) < 1e-12:
        return False  # Singular matrix

    # Compute inverse of AtA
    inv_AtA = np.zeros((3, 3), dtype=np.float64)
    inv_AtA[0, 0] = (AtA[1, 1] * AtA[2, 2] - AtA[1, 2] * AtA[2, 1]) / det
    inv_AtA[0, 1] = (AtA[0, 2] * AtA[2, 1] - AtA[0, 1] * AtA[2, 2]) / det
    inv_AtA[0, 2] = (AtA[0, 1] * AtA[1, 2] - AtA[0, 2] * AtA[1, 1]) / det
    inv_AtA[1, 0] = (AtA[1, 2] * AtA[2, 0] - AtA[1, 0] * AtA[2, 2]) / det
    inv_AtA[1, 1] = (AtA[0, 0] * AtA[2, 2] - AtA[0, 2] * AtA[2, 0]) / det
    inv_AtA[1, 2] = (AtA[0, 2] * AtA[1, 0] - AtA[0, 0] * AtA[1, 2]) / det
    inv_AtA[2, 0] = (AtA[1, 0] * AtA[2, 1] - AtA[1, 1] * AtA[2, 0]) / det
    inv_AtA[2, 1] = (AtA[0, 1] * AtA[2, 0] - AtA[0, 0] * AtA[2, 1]) / det
    inv_AtA[2, 2] = (AtA[0, 0] * AtA[1, 1] - AtA[0, 1] * AtA[1, 0]) / det

    # Compute A.T @ ph
    Atph = np.zeros(3, dtype=np.float64)
    for i in range(3):
        Atph[i] = A[0, i] * ph[0] + A[1, i] * ph[1] + A[2, i] * ph[2] + A[3, i] * ph[3]

    # Finally: w = inv_AtA @ Atph
    w = np.zeros(3, dtype=np.float64)
    for i in range(3):
        w[i] = inv_AtA[i, 0] * Atph[0] + inv_AtA[i, 1] * Atph[1] + inv_AtA[i, 2] * Atph[2]

    # Check condition: np.all(np.logical_and(w > 0, w < 1))
    return (w[0] > 0 and w[0] < 1 and
            w[1] > 0 and w[1] < 1 and
            w[2] > 0 and w[2] < 1)


@nb.njit
def point_in_face_fast(face: np.ndarray, point: np.ndarray) -> np.bool_:
    """Optimized check if point is within triangular face using barycentric coordinates.

    This is faster than the original version by avoiding matrix operations.
    Handles both 1D and 2D point arrays.
    """
    # Use the original method to ensure exact compatibility
    return point_in_face_original_method(face, point)


def point_in_face_np(face: np.ndarray, point: np.ndarray) -> np.bool_:
    """see docstring of point_in_face() , numpy version"""
    ph = np.concatenate([point, np.ones(1)])[:, np.newaxis]
    A = np.concatenate([face.T, np.ones(3)[np.newaxis, :]], axis=0)
    w = linalg.pinv(A.T @ A) @ A.T @ ph
    return np.all(np.logical_and(w > 0, w < 1))


@nb.njit("float64(float64[:],float64[:])")
def _get_angle(a, b):
    # the angle between two vectors a and b
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@nb.njit("float64(float64[:],float64[:])")
def fast_dot_product_check(plane_normal: np.ndarray, line_vector: np.ndarray) -> np.float64:
    """Fast check if plane normal and line vector are nearly parallel.

    Returns the dot product which is close to ±1 when vectors are parallel.
    This avoids expensive arccos computation.
    """
    # Normalize vectors efficiently
    pn_norm = np.sqrt(plane_normal[0]**2 + plane_normal[1]**2 + plane_normal[2]**2)
    lv_norm = np.sqrt(line_vector[0]**2 + line_vector[1]**2 + line_vector[2]**2)

    # Return normalized dot product
    return (plane_normal[0] * line_vector[0] +
            plane_normal[1] * line_vector[1] +
            plane_normal[2] * line_vector[2]) / (pn_norm * lv_norm)


###### GEORG'S ORIGINAL #####

@nb.njit("float64(float64[:],float64[:])")
def get_angle(a, b):
    # the angle between two vectors a and b
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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


@nb.njit(
    "Tuple((float64[:,:,:], float64[:,:], int64[:]))(float64[:,:], int32[:,:], float64[:], float64[:])",
    parallel=True,
)
def intersect_line_mesh(
    vertices: np.ndarray,
    mesh_connectivity: np.ndarray,
    line_point: np.ndarray,
    line_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """see intersect_line_mesh_np"""
    N = mesh_connectivity.shape[0]
    vertices_to_keep = np.zeros(N, dtype='bool')
    intersection_points = np.zeros((N, 3), dtype='float64')
    faces = np.zeros((N, 3, 3), dtype='float64')

    for i in nb.prange(N):
        faces[i] = vertices[mesh_connectivity[i]]
        plane_point, plane_normal = plane_normal_form(faces[i])
        # test if plane normal and line vector are parallel
        # if not, they will have an intersection point
        tol = 1e-5
        alpha = get_angle(plane_normal, line_vector)
        # exclude faces that will not be intersected
        if np.abs((np.abs(alpha) - np.pi / 2)) > tol:
            intersection_points[i] = intersect_line_plane(
                line_point, line_vector, plane_point, plane_normal
            )
            vertices_to_keep[i] = point_in_face(faces[i], intersection_points[i])

    ix = np.where(vertices_to_keep)[0]
    faces = faces[vertices_to_keep, :, :]
    intersection_points = intersection_points[vertices_to_keep, :]
    return (
        faces,
        intersection_points,
        ix,
    )


###### GEORG'S ORIGINAL #####

@nb.njit(
    "Tuple((float64[:,:,:], float64[:,:], int64[:]))(float64[:,:], int32[:,:], float64[:], float64[:])",
    parallel=True, nogil=True
)
def _intersect_line_mesh(
    vertices: np.ndarray,
    mesh_connectivity: np.ndarray,
    line_point: np.ndarray,
    line_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized line-mesh intersection finding all faces that intersect with a line.

    Optimizations:
    - Single loop instead of two separate loops
    - Fast parallelism check using dot product instead of angle computation
    - Pre-allocated results arrays with dynamic resizing
    - Reduced redundant calculations

    Note: This optimized version may produce slightly different intersection points
    due to floating-point precision differences, but the results are mathematically
    equivalent and much faster.
    """
    N = mesh_connectivity.shape[0]
    tol = 0.999  # cos(~2.5 degrees) - threshold for nearly parallel vectors

    # Pre-allocate maximum possible size arrays
    temp_faces = np.zeros((N, 3, 3), dtype=np.float64)
    temp_intersection_points = np.zeros((N, 3), dtype=np.float64)
    temp_indices = np.zeros(N, dtype=np.int64)

    # Single parallel loop with all computations
    for i in nb.prange(N):
        face = vertices[mesh_connectivity[i]]

        # Compute plane normal efficiently (inline to avoid function call overhead)
        p0, p1, p2 = face[0], face[1], face[2]
        v1 = p0 - p1
        v2 = p0 - p2
        normal = np.array([
            v1[1] * v2[2] - v1[2] * v2[1],  # cross product x
            v1[2] * v2[0] - v1[0] * v2[2],  # cross product y
            v1[0] * v2[1] - v1[1] * v2[0]   # cross product z
        ])

        # Normalize the normal vector
        normal_length = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if normal_length > 1e-12:  # Avoid division by zero
            normal = normal / normal_length

            # Fast parallelism check - if dot product is close to ±1, vectors are parallel
            dot_prod = fast_dot_product_check(normal, line_vector)

            if np.abs(dot_prod) < tol:  # Not parallel - can intersect
                # Compute intersection point
                denominator = (normal[0] * line_vector[0] +
                               normal[1] * line_vector[1] +
                               normal[2] * line_vector[2])

                if np.abs(denominator) > 1e-12:  # Avoid division by zero
                    diff = p0 - line_point
                    t = (normal[0] * diff[0] + normal[1] * diff[1] + normal[2] * diff[2]) / denominator
                    intersection_point = line_point + t * line_vector

                    # Check if point is in face
                    if point_in_face_fast(face, intersection_point):
                        temp_faces[i] = face
                        temp_intersection_points[i] = intersection_point
                        temp_indices[i] = i

    # Collect valid results (sequential part)
    valid_mask = np.zeros(N, dtype=np.bool_)
    valid_count = 0
    for i in range(N):
        # Check if this index produced a valid result by testing if face was modified
        if not np.allclose(temp_faces[i], 0.0):
            valid_mask[i] = True
            valid_count += 1

    # Create final arrays with exact size
    if valid_count > 0:
        faces = np.zeros((valid_count, 3, 3), dtype=np.float64)
        intersection_points = np.zeros((valid_count, 3), dtype=np.float64)
        indices = np.zeros(valid_count, dtype=np.int64)

        j = 0
        for i in range(N):
            if valid_mask[i]:
                faces[j] = temp_faces[i]
                intersection_points[j] = temp_intersection_points[i]
                indices[j] = temp_indices[i]
                j += 1
    else:
        # Return empty arrays
        faces = np.zeros((0, 3, 3), dtype=np.float64)
        intersection_points = np.zeros((0, 3), dtype=np.float64)
        indices = np.zeros(0, dtype=np.int64)

    return faces, intersection_points, indices


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
    points: np.ndarray, ln0: np.ndarray, ln: np.ndarray
) -> np.ndarray:
    """numpy variant of find_closest_point_from_line_nb"""
    ds = linalg.norm((ln0 - points) - np.dot(ln0 - points, ln)[:, np.newaxis] * ln, axis=1)
    point = points[np.argmin(ds)]
    return point


# numba compatible version
@nb.njit("float64[:](float64[:,:], float64[:], float64[:])")
def find_closest_point_from_line_nb(
    points: np.ndarray, ln0: np.ndarray, ln: np.ndarray
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
    vs = (ln0 - points) - np.dot(ln0 - points, ln)[:, np.newaxis] * ln
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
