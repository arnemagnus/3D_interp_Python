# Cython header file for the trilinear interpolator extension module

# NumPy is just about essential regarding scientific computing in Python, and
# well integrated with Cython;
import numpy as np
cimport numpy as np


# The C math library contains lots of useful functions. Using these is much more
# efficient than calling e.g. NumPy's implementation, when working at C level.
# Hence:
from libc.math cimport fmod as c_fmod, floor as c_floor, pow as c_pow


cdef class TrilinearInterpolator:
    cdef:
        # The boundaries of the physical domain
        # (needed in order to properly transform input coordinates to
        # the relevant interpolation voxel)
        double x_min, x_max, y_min, y_max, z_min, z_max
        # The number of points along each axis
        int nx, ny, nz
        # The grid spacings are needed for the finite difference approximation
        # of derivatives, in addition to transform input coordinates to
        # the relevant interpolation voxel
        double dx, dy, dz
        # The 3D-data to be interpolated
        double[:,:,::1] data
        # Boolean flag as to whether or not periodic BC should be enforced
        bint periodic

    # A C level function which evaluates the interpolated function in a single
    # point
    cdef double _ev_(self, double x, double y, double z)

    # A C level function which evaluates the interpolated function on the grid
    # spanned by the one-dimensional input arrays x, y and z
    cdef np.ndarray[np.float64_t,ndim=3] _ev_grid_(self, double[::1] x,
                                                         double[::1] y,
                                                         double[::1] z)


    # A C level function which finds the indices of the interpolation voxel
    # of interest, subject to periodic boundary conditions
    cdef _set_periodic_voxel_indices_(self, double *x, double *y, double *z,
                                            int *ix, int *ixp1,
                                            int *iy, int *iyp1,
                                            int *iz, int *izp1)

    # A C level function which finds the indices of the interpolation voxel
    # of interest, when just pure interpolation is allowed
    cdef _set_nonperiodic_voxel_indices_(self, double *x, double *y, double *z,
                                            int *ix, int *ixp1,
                                            int *iy, int *iyp1,
                                            int *iz, int *izp1)
