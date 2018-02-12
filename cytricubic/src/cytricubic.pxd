# Cython header file for the local tricubic polynomial interpolator extension
# module

# NumPy is just about essential regarding scientific computing in Python, and
# well integrated with Cython;
import numpy as np
cimport numpy as np

# The C math library contains lots of useful functions. Using these is much more
# efficient than calling e.g. NumPy's implementation, when working at C level.
# Hence:
from libc.math cimport fmod as c_fmod, floor as c_floor, pow as c_pow

# When evaluating the interpolated function, one generally needs to multiply
# a 64-by-64 matrix with a 64-by-1 vector in order to obtain correct
# interpolation coefficients. In order to do this efficiently, we use the
# BLAS level two function 'dgemv':
from scipy.linalg.cython_blas cimport dgemv as cy_dgemv

# The attached header file "coeff_.h" contains the int (equiv. to
# numpy.int32) representation of the 64-by-64 matrix defining the linear system
# used for the three-dimensional interpolation.
#
cdef extern from "../include/coeff_.h":
    int get_coeff(int*, int*)
#
# A simple wrapper function, aptly named "get_coeff", extracts the individual
# matrix elements, which is needed in order to set the system matrix elements
# for an instance of the TricubicInterpolator class. Explicitly typing
# the 64-times-64 coefficients by hand is out of the question.


cdef class TricubicInterpolator:
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
        # The 64-by-64-matrix which defines the linear 3D interpolation system
        double A[64][64]
        # Container for the intermediate coefficients needed to compute
        # interpolation coefficients within a given voxel
        double psi[64]
        # Container for the interpolation coefficients within a given voxel
        double coeffs[64]
        # Flag indicating whether or not the interpolator has previously been
        # calibrated for any voxel
        bint calibrated
        # Indices keeping track of which voxel the interpolator calibration
        # was most recently performed for
        int xi, yi, zi

    # A C level function which evaluates the interpolated function (or its
    # derivatives) in a single point
    cdef double _ev_(self, double x, double y, double z, int kx, int ky, int kz)

    # A convenience function, used to transform (tuples of) integers to
    # a single index for the interpolation coefficient array
    cdef int _ind_(self, int i, int j, int k)

    # A C level function which evaluates the interpolated function (or its
    # derivatives) on the grid spanned by the one-dimensional input arrays x, y
    # and z
    cdef np.ndarray[np.float64_t,ndim=3] _ev_grid_(self, double[::1] x,
                                                         double[::1] y,
                                                         double[::1] z,
                                                         int kx, int ky, int kz)


    # A C level function which computes the interpolation coefficients within
    # the voxel given by the indices of its reference corner, subject to
    # periodic boundary conditions:
    cdef _calibrate_periodic_(self, int xi, int yi, int zi)

    # A C level function which finds the indices of the interpolation voxel
    # of interest, subject to periodic boundary conditions
    cdef _set_periodic_voxel_indices_(self,
                                    int ix, int *ixm1, int *ixp1, int *ixp2,
                                    int iy, int *iym1, int *iyp1, int *iyp2,
                                    int iz, int *izm1, int *izp1, int *izp2)

    # A C level function which inserts the function values at the corners of
    # the interpolation voxel in the right place in the intermediate container
    cdef _set_periodic_vals_(self, int x, int xp1,
                                   int y, int yp1,
                                   int z, int zp1)

    # A C level function which inserts centered difference approximations of the
    # first derivatives at the corners of the interpolation voxel in the right
    # place in the intermediate container, subject to periodic boundary
    # conditions
    cdef _set_periodic_derivs_(self, int x, int xm1, int xp1, int xp2,
                                     int y, int ym1, int yp1, int yp2,
                                     int z, int zm1, int zp1, int zp2)

    # A C level function which inserts centered difference approximations of the
    # mixed second derivatives at the corners of the interpolation voxel in the
    # right place in the intermediate container, subject to periodic boundary
    # conditions
    cdef _set_periodic_mxd_2derivs_(self, int x, int xm1, int xp1, int xp2,
                                          int y, int ym1, int yp1, int yp2,
                                          int z, int zm1, int zp1, int zp2)

    # A C level function which inserts centered difference approximations of the
    # mixed third derivative d3f/dxdydz at the corners of the interpolation
    # voxel in the right place in the intermediate container, subject to
    # periodic boundary conditions
    cdef _set_periodic_mxd_3deriv_(self, int x, int xm1, int xp1, int xp2,
                                         int y, int ym1, int yp1, int yp2,
                                         int z, int zm1, int zp1, int zp2)

    # A C level function which computes the interpolation coefficients within
    # the voxel given by the indices of its reference corner, when periodic
    # boundary conditions are not enforced:
    cdef _calibrate_nonperiodic_(self, int xi, int yi, int zi)

    # A C level function which inserts the function values at the corners of
    # the interpolation voxel in the right place in the intermediate container
    cdef _set_nonperiodic_vals_(self, int x, int y, int z)

    # A C level function which inserts finite difference approximations of the
    # first derivatives at the corners of the interpolation voxel in the right
    # place in the intermediate container, when periodic boundary conditions
    # are not enforced
    cdef _set_nonperiodic_derivs_(self, int x, int y, int z)

    # Computing the derivatives when periodic boundary conditions are not
    # enforced is sufficiently convoluted to warrant setting each
    # component separately by its own function. Hence
    cdef _set_nonperiodic_dfdx_(self, int x, int y, int z)
    cdef _set_nonperiodic_dfdy_(self, int x, int y, int z)
    cdef _set_nonperiodic_dfdz_(self, int x, int y, int z)

    # A C level function which inserts finite difference approximations of the
    # mixed second derivatives at the corners of the interpolation voxel in the
    # right place in the intermediate container, when periodic boundary
    # conditions are not enforced
    cdef _set_nonperiodic_mxd_2derivs_(self, int x, int y, int z)

    # Computing the mixed second derivatives when periodic boundary conditions
    # are not enforced is sufficiently convoluted to warrant setting each
    # component separately by its own function. Hence
    cdef _set_nonperiodic_d2fdxdy_(self, int x, int y, int z)
    cdef _set_nonperiodic_d2fdxdz_(self, int x, int y, int z)
    cdef _set_nonperiodic_d2fdydz_(self, int x, int y, int z)

    # A C level function which inserts finite difference approximations of the
    # mixed third derivative at the corners of the interpolation voxel in the
    # right place in the intermediate container, when periodic boundary
    # conditions are not enforced
    cdef _set_nonperiodic_mxd_3deriv_(self, int x, int y, int z)

    # A C level function which computes the interpolation coefficients
    # within a given voxel, making use of the BLAS level two function 'dgemv':
    cdef _compute_coeffs_by_blas_dgemv_(self)
