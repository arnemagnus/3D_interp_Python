#!python
#cython: language_level=3, embedsignature=True

"""
This module contains an implementation of a tricubic interpolation routine
in 3D, the theoretical foundation of which is found in

    Lekien, F and Marsden, J (2005):
        'Tricubic Interpolation in Three Dimensions',
            in Journal of Numerical Methods and Engineering(63), pp. 455-471,
            available at
		http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.89.7835
	    (checked February 5th, 2018)

The interpolation assumes periodic boundary conditions along all three
abscissae.
"""

# NumPy is just about essential regarding scientific computing in Python,
# and well integrated with Cython;
import numpy as np
cimport numpy as np

# The cython library contains a lot of useful functionalities, such as
# compiler flags;
cimport cython

# When evaluating the interpolated function, one generally needs to multiply
# a 64-by-64 matrix with a 64-by-1 vector in order to obtain correct
# interpolation coefficients. In order to do this efficiently, we use
# the BLAS level two function 'dgemv':
from scipy.linalg.cython_blas cimport dgemv as cy_dgemv

# The C math library contains lots of useful functions. Using these is much
# more efficient than calling e.g. NumPy's implementation, when working at C
# level. Hence:
from libc.math cimport fmod as c_fmod, floor as c_floor, pow as c_pow

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

# Here follows the implementation of the Cython extension type:
cdef class TricubicInterpolator:
    """
    This class provides a Python object interface to optimized C code which
    enables tricubic interpolation in three dimensions, where periodic boundary
    conditions in all three dimensions is implicitly assumed.

    This particular implementation guarantees that the interpolated object
    has continuous first derivatives, *mixed* second derivatives (i.e.,
    d2f/dxdy, d2f/dxdz and d2f/dydz) in addition to d3f/dxdydz.

    In *some* cases, the other second derivatives, that is, d2f/dx2, d2f/dy2
    and d2f/dz2, may be continuous. The same applies to the third derivatives.
    This depends strongly on the smoothness of the *actual* function.
    """
    # All extension type attributes must be pre-declared at compile time.
    # Typed attributes are, by default, only accessible from Cython.
    cdef:
        # The minimum boundaries of the computational domain
        # (needed in order to properly transform input coordinates to
        # the relevant interpolation voxel)
        double x_min, y_min, z_min
        # The number of points along each axis
        int nx, ny, nz
        # The grid spacings are needed for the finite difference approximation
        # of derivatives, in addition to transform input coordinates to
        # the relevant interpolation voxel
        double dx, dy, dz
        # The 3D-data to be interpolated
        double[:,:,::1] data
        # The 64-by-64 matrix which defines the linear 3D interpolation system
        double A[64][64]
        # Container for the intermediate coefficients needed to compute
        # interpolation coefficients within a given voxel
        double psi[64]
        # Container for the interpolation coefficients within a given voxel
        double coeffs[64]
        # Flag indicating whether or not the interpolator has been calibrated
        # for any voxel
        bint calibrated
        # Indices keeping track of which voxel the interpolator calibration
        # was most recently performed for
        int xi, yi, zi


    # The following Cython compilation flags turn off the Pythonic
    # bounds check and wraparound functionality with regards to
    # array indexing.
    #
    # These safeguards are generally nice to have, but drastically decrease
    # performance. So, by being very explicit regarding indexing, we can
    # turn them off - at the risk of segfaults (best case scenario) or
    # corrupted data (worst case scenario), should we make an error.
    #
    # In short: These should be used with caution.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self,double[::1] x not None,double[::1] y not None,
            double[::1] z not None,double[:,:,::1] data not None):
        """
        TricubicInterpolator(x, y, z, data)

        Constructor for a TricubicInterpolator object. Intended for use on
        a Cartesian, three-dimensional grid of rectangular parallelepipeds.
        The grid spacings need not be the same along any pair of axes.

        param: x    -- A 1D numpy array of np.float64, defining the coordinates
                       along the first axis, at which the function has been
                       sampled. Must be strictly increasing.
                       *IMPORTANT*: len(x) >= 4.
        param: x    -- A 1D numpy array of np.float64, defining the coordinates
                       along the second axis, at which the function has been
                       sampled. Must be strictly increasing.
                       *IMPORTANT*: len(y) >= 4.
        param: z    -- A 1D numpy array of np.float64, defining the coordinates
                       along the third axis, at which the function has been
                       sampled. Must be strictly increasing.
                       *IMPORTANT*: len(z) >= 4.
        param: data -- A 3D numpy array of np.float64, containing the sampled
                       function values f(x,y,z) on the grid spanned by x, y
                       and z. Shape: (len(x),len(y),len(z)).
        """
        # Local variables:
        cdef:
            int i, j # Loop counters

        if(x.shape[0] < 4 or y.shape[0] < 4 or z.shape[0] < 4):
            raise RuntimeError("In order to perform tricubic interpolation,\
                        all abscissa arrays must contain at least four elements!")

        if(x.shape[0] != data.shape[0] or y.shape[0] != data.shape[1]
                or z.shape[0] != data.shape[2]):
            raise RuntimeError("Input data not properly aligned. See\
                                constructor docstring for details.")

        # Store minimum boundaries of physical domain
        self.x_min = x[0]
        self.y_min = y[0]
        self.z_min = z[0]
        # Store physical grid spacing
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0]
        # Store number of elements along each abscissa axis
        self.nx = x.shape[0]-1
        self.ny = y.shape[0]-1
        self.nz = z.shape[0]-1
        # Store function values to interpolate
        self.data = data[:self.nx-1,:self.ny-1,:self.nz-1]
        # Before the first evaluation, the interpolator object is
        # uncalibrated
        self.calibrated = 0
        # Explicitly set each element of the matrix A, using the predefined
        # 64-by-64 matrix in the helper file coeff_.h
        for j in range(64):
            for i in range(64):
                self.A[i][j] = get_coeff(&i,&j)

    # A custom, thin Python wrapper for the _ev_ function, defined at C level;
    def ev(self,double x, double y, double z,
            int kx = 0, int ky = 0, int kz = 0):
        """
        TricubicInterpolator.ev(x,y,z,kx,ky,kz)

        Evaluate the interpolated function, or its derivatives,  at a single
        point.

        param: x   -- Double-precision coordinate along the x axis.
        param: y   -- Double-precision coordinate along the y axis.
        param: z   -- Double-precision coordinate along the z axis.
        OPTIONAL:
        param: kx -- Integer specifying the order of the partial derivative
                        along the x axis. 0 <= kx <= 3. Default: kx = 0.
        param: ky  -- Integer specifying the order of the partial derivative
                        along the y axis. 0 <= ky <= 3. Default: ky = 0.
        param: kz  -- Integer specifying the order of the partial derivative
                        along the z axis. 0 <= kz <= 3. Default: kz = 0.
        """
        return self._ev_(x,y,z,kx,ky,kz)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _ev_(self,double x, double y, double z,
                int kx, int ky, int kz):
        cdef:
            double res = 0.
            int x_ind, y_ind, z_ind
            int i, j, k, ijk = 0
            double *coeffs = self.coeffs

        if(kz < 0 or ky < 0 or kz < 0):
            raise RuntimeError("Derivative order must be nonnegative.")

        if(kx > 3 or ky > 3 or kz > 3):
            raise RuntimeError("Derivative order can't be larger than 3.")

	# Determine the relative coordinates of the point in question, within
	# its voxel
        x = c_fmod((x-self.x_min)/self.dx,self.nx)
        y = c_fmod((y-self.y_min)/self.dy,self.ny)
        z = c_fmod((z-self.z_min)/self.dz,self.nz)

	# Enforce periodic BC
        while(x < 0):
            x += self.nx
        while(y < 0):
            y += self.ny
        while(z < 0):
            z += self.nz

        x_ind = int(c_floor(x))
        y_ind = int(c_floor(y))
        z_ind = int(c_floor(z))

        if(self.calibrated == 0 or x_ind != self.xi or y_ind != self.yi or z_ind != self.zi):
            self._calibrate_(x_ind,y_ind,z_ind)

        x -= x_ind
        y -= y_ind
        z -= z_ind

        cdef:
            double cont
            int w

        for k in range(kz,4):
            for j in range(ky,4):
                for i in range(kx,4):
                    cont = coeffs[self.ijk2n(i,j,k)]*c_pow(x,i-kx)*c_pow(y,j-ky)*c_pow(z,k-kz)
                    for w in range(kx):
                        cont *= (i-w)
                    for w in range(ky):
                        cont *= (j-w)
                    for w in range(kz):
                        cont *= (k-w)
                    res += cont
        return res/(c_pow(self.dx,kx)*c_pow(self.dy,ky)*c_pow(self.dz,kz))


    cdef int ijk2n(self,int i, int j, int k):
        return(i+4*j+16*k)


    def ev_grid(self,double[:,:,::1] x, double[:,:,::1] y, double[:,:,::1] z,
                    int kx = 0, int ky = 0, int kz = 0):
        return self._ev_grid_(x,y,z,kx,ky,kz)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _ev_grid_(self,double[:,:,::1] x, double[:,:,::1] y, double[:,:,::1] z,
                int kx, int ky, int kz):
        cdef:
            int i, j, k
            int x_sh0 = x.shape[0], x_sh1 = x.shape[1], x_sh2 = x.shape[2]
            int y_sh0 = y.shape[0], y_sh1 = y.shape[1], y_sh2 = y.shape[2]
            int z_sh0 = z.shape[0], z_sh1 = z.shape[1], z_sh2 = z.shape[2]

        if(x_sh0 != y_sh0 or x_sh0 != z_sh0 or y_sh0 != z_sh0
                or x_sh1 != y_sh1 or x_sh1 != z_sh1 or y_sh1 != z_sh1
                or x_sh2 != y_sh2 or x_sh2 != z_sh2 or y_sh2 != z_sh2):
            raise RuntimeError("Array dimensions inconsistent!")

        cdef np.ndarray[np.float64_t,ndim=3] res = np.empty((x_sh0,
                                                             x_sh1,
                                                             x_sh2),
                                                            dtype=np.float64)

        for k in range(x_sh2):
            for j in range(x_sh1):
                for i in range(x_sh0):
                    res[i,j,k] = self._ev_(x[i,j,k],y[i,j,k],z[i,j,k],
                                            kx,ky,kz)

        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _calibrate_(self,int x, int y, int z):
        cdef:
            int xm1, xp1, xp2
            int ym1, yp1, yp2
            int zm1, zp1, zp2

        # Precompute the voxel indices plus/minus one and two, in order to
        # explicitly provide the Python array wraparound and bounds check
        # feature, so that we can disable it at a Python level (to enhance
        # performance)
        xm1 = (x-1)%self.nx
        xp1 = (x+1)%self.nx
        xp2 = (x+2)%self.nx

        ym1 = (y-1)%self.ny
        yp1 = (y+1)%self.ny
        yp2 = (y+2)%self.ny

        zm1 = (z-1)%self.nz
        zp1 = (z+1)%self.nz
        zp2 = (z+2)%self.nz

        if(xm1 < 0):
            xm1 += self.nx
        if(xp1 < 0):
            xp1 += self.nx
        if(xp2 < 0):
            xp2 += self.nx

        if(ym1 < 0):
            ym1 += self.ny
        if(yp1 < 0):
            yp1 += self.ny
        if(yp2 < 0):
            yp2 += self.ny

        if(zm1 < 0):
            zm1 += self.nz
        if(zp1 < 0):
            zp1 += self.nz
        if(zp2 < 0):
            zp2 += self.nz

        # Values of f(x,y,z) at the corners of the voxel
        self._set_values_(x,xp1,y,yp1,z,zp1)
        # First derivatives of f(x,y,z) at the corners of the voxel
        self._set_first_derivatives_(x,xm1,xp1,xp2,
                                        y,ym1,yp1,yp2,
                                        z,zm1,zp1,zp2)
        # Mixed second derivatives of f(x,y,z) at the corners of the voxel
        self._set_second_drvtvs_(x,xm1,xp1,xp2,
                                 y,ym1,yp1,yp2,
                                 z,zm1,zp1,zp2)
        # Values of d3f/dxdydz at the corners of the voxel
        self._set_third_drvtv_(x,xm1,xp1,xp2,
                               y,ym1,yp1,yp2,
                               z,zm1,zp1,zp2)
        # Convert voxel values and partial derivatives to interpolation
        # coefficients
        self._solve_by_blas_dgemv_()
        # Remember the configuration for the next call
        self.xi, self.yi, self.zi = x, y, z
        self.calibrated = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_values_(self,int x,int xp1,int y,int yp1,int z,int zp1):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data = self.data
        # Values of f(x,y,z) at the corners of the voxel
        psi[0]  = data[x,y,z]
        psi[1]  = data[xp1,y,z]
        psi[2]  = data[x,yp1,z]
        psi[3]  = data[xp1,yp1,z]
        psi[4]  = data[x,y,zp1]
        psi[5]  = data[xp1,y,zp1]
        psi[6]  = data[x,yp1,zp1]
        psi[7]  = data[xp1,yp1,zp1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_first_derivatives_(self,int x,int xm1,int xp1,int xp2,
                                        int y, int ym1, int yp1, int yp2,
                                        int z, int zm1, int zp1, int zp2):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data = self.data
        # Values of df/dx at the corners of the voxel
        psi[8]  = 0.5*(data[xp1,y,z]-data[xm1,y,z])
        psi[9]  = 0.5*(data[xp2,y,z]-data[x,y,z])
        psi[10] = 0.5*(data[xp1,yp1,z]-data[xm1,yp1,z])
        psi[11] = 0.5*(data[xp2,yp1,z]-data[x,yp1,z])
        psi[12] = 0.5*(data[xp1,y,zp1]-data[xm1,y,zp1])
        psi[13] = 0.5*(data[xp2,y,zp1]-data[x,y,zp1])
        psi[14] = 0.5*(data[xp1,yp1,zp1]-data[xm1,yp1,zp1])
        psi[15] = 0.5*(data[xp2,yp1,zp1]-data[x,yp1,zp1])
        # Values of df/dy at the corners of the voxel
        psi[16] = 0.5*(data[x,yp1,z]-data[x,ym1,z])
        psi[17] = 0.5*(data[xp1,yp1,z]-data[xp1,ym1,z])
        psi[18] = 0.5*(data[x,yp2,z]-data[x,y,z])
        psi[19] = 0.5*(data[xp1,yp2,z]-data[xp1,y,z])
        psi[20] = 0.5*(data[x,yp1,zp1]-data[x,ym1,zp1])
        psi[21] = 0.5*(data[xp1,yp1,zp1]-data[xp1,ym1,zp1])
        psi[22] = 0.5*(data[x,yp2,zp1]-data[x,y,zp1])
        psi[23] = 0.5*(data[xp1,yp2,zp1]-data[xp1,y,zp1])
        # Values of df/dz at the corners of the voxel
        psi[24] = 0.5*(data[x,y,zp1]-data[x,y,zm1])
        psi[25] = 0.5*(data[xp1,y,zp1]-data[xp1,y,zm1])
        psi[26] = 0.5*(data[x,yp1,zp1]-data[x,yp1,zm1])
        psi[27] = 0.5*(data[xp1,yp1,zp1]-data[xp1,yp1,zm1])
        psi[28] = 0.5*(data[x,y,zp2]-data[x,y,z])
        psi[29] = 0.5*(data[xp1,y,zp2]-data[xp1,y,z])
        psi[30] = 0.5*(data[x,yp1,zp2]-data[x,yp1,z])
        psi[31] = 0.5*(data[xp1,yp1,zp2]-data[xp1,yp1,z])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_second_drvtvs_(self,int x,int xm1,int xp1,int xp2,
                                   int y,int ym1,int yp1,int yp2,
                                   int z,int zm1,int zp1,int zp2):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data = self.data
        # Values of d2f/dxdy at the corners of the voxel
        psi[32] = 0.25*((data[xp1,yp1,z]-data[xm1,yp1,z])
                            - (data[xp1,ym1,z]-data[xm1,ym1,z]))
        psi[33] = 0.25*((data[xp2,yp1,z]-data[x,yp1,z])
                            -(data[xp2,ym1,z]-data[x,ym1,z]))
        psi[34] = 0.25*((data[xp1,yp2,z]-data[xm1,y,z])
                            -(data[xp1,y,z]-data[xm1,y,z]))
        psi[35] = 0.25*((data[xp2,yp2,z]-data[x,yp2,z])
                            -(data[xp2,y,z]-data[x,y,z]))
        psi[36] = 0.25*((data[xp1,yp1,zp1]-data[xm1,yp1,zp1])
                            -(data[xp1,ym1,zp1]-data[xm1,ym1,zp1]))
        psi[37] = 0.25*((data[xp2,yp1,zp1]-data[x,yp1,zp1])
                            -(data[xp2,ym1,zp1]-data[x,ym1,zp1]))
        psi[38] = 0.25*((data[xp1,yp2,zp1]-data[xm1,yp2,zp1])
                            -(data[xp1,y,zp1]-data[xm1,y,zp1]))
        psi[39] = 0.25*((data[xp2,yp2,zp1]-data[x,yp2,zp1])
                            -(data[xp2,y,zp1]-data[x,y,zp1]))
        # Values of d2f/dxdz at the corners of the voxel
        psi[40] = 0.25*((data[xp1,y,zp1]-data[xm1,y,zp1])
                            -(data[xp1,y,zm1]-data[xm1,y,zm1]))
        psi[41] = 0.25*((data[xp2,y,zp1]-data[x,y,zp1])
                            -(data[xp2,y,zm1]-data[x,y,zm1]))
        psi[42] = 0.25*((data[xp1,yp1,zp1]-data[xm1,yp1,zp1])
                            -(data[xp1,yp1,zm1]-data[xm1,yp1,zm1]))
        psi[43] = 0.25*((data[xp2,yp1,zp1]-data[x,yp1,zp1])
                            -(data[xp2,yp1,zm1]-data[x,yp1,zm1]))
        psi[44] = 0.25*((data[xp1,y,zp2])-data[xm1,y,zp2]
                            -(data[xp1,y,z]-data[xm1,y,z]))
        psi[45] = 0.25*((data[xp2,y,zp2]-data[x,y,zp2])
                            -(data[xp2,y,z]-data[x,y,z]))
        psi[46] = 0.25*((data[xp1,yp2,zp2]-data[xm1,yp2,zp2])
                            -(data[xp1,yp2,z]-data[xm1,yp2,z]))
        psi[47] = 0.25*((data[xp2,yp2,zp2]-data[x,yp2,zp2])
                            -(data[xp2,yp2,z]-data[x,yp2,z]))
        # Values of d2f/dydz at the corners of the voxel
        psi[48] = 0.25*((data[x,yp1,zp1]-data[x,ym1,zp1])
                            -(data[x,yp1,zm1]-data[x,ym1,zm1]))
        psi[49] = 0.25*((data[xp1,yp1,zp1]-data[xp1,ym1,zp1])
                            -(data[xp1,yp1,zm1]-data[xp1,ym1,zm1]))
        psi[50] = 0.25*((data[x,yp1,zp1]-data[x,ym1,zp1])
                            -(data[x,yp1,zm1]-data[x,ym1,zm1]))
        psi[51] = 0.25*((data[xp1,yp2,zp1]-data[xp1,y,zp1])
                            -(data[xp1,yp2,zm1]-data[xp1,y,zm1]))
        psi[52] = 0.25*((data[x,yp1,zp2]-data[x,ym1,zp2])
                            -(data[x,yp1,z]-data[x,ym1,z]))
        psi[53] = 0.25*((data[xp1,yp1,zp2]-data[xp1,ym1,zp2])
                            -(data[xp1,yp1,z]-data[xp1,ym1,z]))
        psi[54] = 0.25*((data[x,yp2,zp2]-data[x,y,zp2])
                            -(data[x,yp2,z]-data[x,y,z]))
        psi[55] = 0.25*((data[xp2,yp2,zp2]-data[xp2,y,zp2])
                            -(data[xp2,yp2,z]-data[xp2,y,z]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_third_drvtv_(self,int x,int xm1,int xp1,int xp2,
                                int y,int ym1,int yp1,int yp2,
                                int z,int zm1,int zp1,int zp2):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data = self.data
        # Values of d3f/dxdydz at the corners of the voxel
        psi[56] = 0.125*(((data[xp1,yp1,zp1]-data[xm1,yp1,zp1])
                                -(data[xp1,ym1,zp1]-data[xm1,ym1,zp1]))
                            -((data[xp1,yp1,zm1]-data[xm1,yp1,zm1])
                                -(data[xp1,ym1,zm1]-data[xm1,ym1,zm1])))
        psi[57] = 0.125*(((data[xp2,yp1,zp1]-data[x,yp1,zp1])
                                -(data[xp2,ym1,zp1]-data[x,ym1,zp1]))
                            -((data[xp2,yp1,zm1]-data[x,yp1,zm1])
                                -(data[xp2,ym1,zm1]-data[x,ym1,zm1])))
        psi[58] = 0.125*(((data[xp1,yp2,zp1]-data[xm1,yp2,zp1])
                                -(data[xp1,y,zp1]-data[xm1,y,zp1]))
                            -((data[xp1,yp2,zm1]-data[xm1,yp2,zm1])
                                -(data[xp1,y,zm1]-data[xm1,y,zm1])))
        psi[59] = 0.125*(((data[xp2,yp2,zp1]-data[x,yp2,zp1])
                                -(data[xp2,y,zp1]-data[x,y,zp1]))
                            -((data[xp2,yp2,zm1]-data[x,yp2,zm1])
                                -(data[xp2,y,zm1]-data[x,y,zm1])))
        psi[60] = 0.125*(((data[xp1,yp1,zp2]-data[xm1,yp1,zp2])
                                -(data[xp1,ym1,zp2]-data[xm1,ym1,zp2]))
                            -((data[xp1,yp1,z]-data[xm1,yp1,z])
                                -(data[xp1,ym1,z]-data[xm1,ym1,z])))
        psi[61] = 0.125*(((data[xp2,yp1,zp2]-data[x,yp1,zp2])
                                -(data[xp2,ym1,zp2]-data[x,ym1,zp2]))
                            -((data[xp2,yp1,z]-data[x,yp1,z])
                                -(data[xp2,ym1,z]-data[x,ym1,z])))
        psi[62] = 0.125*(((data[xp1,yp2,zp2]-data[xm1,yp2,zp2])
                                -(data[xp1,y,zp2]-data[xm1,y,zp2]))
                            -((data[xp1,yp2,z]-data[xm1,y,z])
                                -(data[xp1,y,z]-data[xm1,y,z])))
        psi[63] = 0.125*(((data[xp2,yp2,zp2]-data[x,yp2,zp2])
                                -(data[xp2,y,zp2]-data[x,y,zp2]))
                            -((data[xp2,yp2,z]-data[x,yp2,z])
                                -(data[xp2,y,z]-data[x,y,z])))

    cdef _solve_by_blas_dgemv_(self):
        # Computes matrix-vector product needed to identify interpolation
        # coefficients within a given voxel.
        #
        # Does so by calling the BLAS level 2 function 'dgemv', included
        # from within the SciPy linear algebra library.
        #
        # Detailed documentation is available at e.g.
        # 	http://www.netlib.org/lapack/explore-html/dc/da8/dgemv_8f.html
        #	(checked Feb. 7, 2018)
        #
        # Information regarding the BLAS functions that are callable from
        # Cython, making use of the SciPy linear algebra library, can be found
        # at e.g.
        #       https://docs.scipy.org/doc/scipy/reference/linalg.cython_blas.html
        #       (checked Feb. 7, 2018)
        #
        # 'dgemv' performs one of the matrix-vector operations
        #       y := alpha*A*x + beta*y
        # or
        #       y := alpha*A**T*x + beta*y
        # where alpha and beta are double-precision scalars, x and y are
        # double-precision (M- and N-)vectors, and A is a double-precision,
        # M-by-N matrix.
        #
        # For our purposes, we want to compute
        #       self.coeffs := self.A*self.psi
        # which is reflected in the definitions of the input variables to
        # 'dgemv', in the following:
        cdef:
            # The BLAS function requires data to be stored in Fortran-contiguous
            # order. The simplest way to transform the system matrix, which
            # is stored in C-contiguous order, to the required form, is by
            # letting the low level BLAS routine perform matrix transposition
            # for us. Hence:
            char* trans = 'T'
            # Other options: trans = 'N' -> No matrix transposition
            #                trans = 'C' -> Conjugate transpose (equivalent to
            #                               'T' when working with real numbers)
            #
            # 'dgemv' needs to know the number of rows (M) and columns (N)
            # of the matrix A:
            int M = 64, N = 64
            # For our purposes, alpha = 1 and beta = 0:
            double alpha = 1, beta = 0
            # Leading dimension of A: The number of rows in A
            int LDA = 64
            # Increment in X: Increment for the elements of X (for our purposes,
            #                   X = self.psi)
            # Increment in Y: Increment for the elements of Y (for our purposes,
            #                   Y = self.coeffs)
            int INCX = 1, INCY = 1

        # The function 'dgemv' from the Cython interface to the SciPy linear
        # algebra library takes Fortran-style pointer arguments,
        # leaving the actual function call somewhat convoluted:
        cy_dgemv(trans, # Already a pointer
                 &M,
                 &N,
                 &alpha,
                 &self.A[0][0],
                 &LDA,
                 &self.psi[0],
                 &INCX,
                 &beta,
                 &self.coeffs[0],
                 &INCY)
