import numpy as np


def cart2sph(x, y, z):
    """ Converts cartesian to spherical Coordinates"""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x) * 180 / np.pi
    phi = np.arccos(z / r) * 180 / np.pi
    phi[np.isnan(phi)] = 0
    return r, theta, phi


def sph2cart(r, theta, phi):
    """ Converts Spherical to Cartesian coordinates"""
    x = r * np.cos(theta / 180 * np.pi) * np.sin(phi / 180 * np.pi)
    y = r * np.sin(theta / 180 * np.pi) * np.sin(phi / 180 * np.pi)
    z = r * np.cos(phi / 180 * np.pi)
    return x, y, z


class BrainCoordinates:
    """
    Class for mapping and indexing a 3D array to real-world coordinates
    x = ml, right positive
    y = ap, anterior positive
    z = dv, dorsal positive

    The layout of the Atlas dimension is done according to the most used sections so they lay
    contiguous on disk assuming C-ordering: V[iap, iml, idv]

    nxyz: number of elements along each cartesian axis (nx, ny, nz) = (nml, nap, ndv)
    xyz0: coordinates of the element volume[0, 0, 0]] in the coordinate space
    dxyz: spatial interval of the volume along the 3 dimensions
    """

    def __init__(self, nxyz, xyz0=[0, 0, 0], dxyz=[1, 1, 1]):
        if np.isscalar(dxyz):
            dxyz = [dxyz for i in range(3)]
        self.x0, self.y0, self.z0 = xyz0
        self.dx, self.dy, self.dz = dxyz
        self.nx, self.ny, self.nz = nxyz

    """Methods distance to indice"""
    def x2i(self, x):
        return (x - self.x0) / self.dx

    def y2i(self, y):
        return (y - self.y0) / self.dy

    def z2i(self, z):
        return (z - self.z0) / self.dz

    def xyz2i(self, xyz):
        return np.array([self.x2i(xyz[0]), self.y2i(xyz[1]), self.z2i(xyz[2])])

    """Methods indices to distance"""
    def i2x(self, ind):
        return ind * self.dx + self.x0

    def i2y(self, ind):
        return ind * self.dy + self.y0

    def i2z(self, ind):
        return ind * self.dz + self.z0

    """Methods bounds"""
    @property
    def xlim(self):
        return self.i2x(np.array([0, self.nx - 1]))

    @property
    def ylim(self):
        return self.i2y(np.array([0, self.ny - 1]))

    @property
    def zlim(self):
        return self.i2z(np.array([0, self.nz - 1]))
