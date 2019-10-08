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
    contiguous on disk

    vshape: shape of 3D volume
    xyz0: coordinates of the element volume[0, 0, 0]] in the coordinate space
    dxyz: spatial interval of the volume along the 3 dimensions
    """

    def __init__(self, vshape, xyz0=[0, 0, 0], dxyz=[1, 1, 1]):
        self.x0, self.y0, self.z0 = xyz0
        self.dx, self.dy, self.dz = dxyz
        self.nx = vshape[1]
        self.ny = vshape[0]
        self.nz = vshape[2]

    """Methods distance to indice"""
    def x2i(self, x):
        return (x - self.x0) / self.dx

    def y2i(self, y):
        return (y - self.y0) / self.dy

    def z2i(self, z):
        return (z - self.z0) / self.dz

    """Methods indices to distance"""
    def i2x(self, ind):
        return ind * self.dx + self.x0

    def i2y(self, ind):
        return ind * self.dy + self.y0

    def i2z(self, ind):
        return ind * self.dz + self.z0
