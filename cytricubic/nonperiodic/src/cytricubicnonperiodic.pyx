"""
This module contains an object oriented implementation of a local tricubic
polynomial interpolation routine in 3D, the theoretical foundation of which is
found in

    Lekien, F and Marsden, J (2005):
        'Tricubic Interpolation in Three Dimensions',
        in Journal of Numerical Methods and Engineering(63), pp. 455-471,
	doi:10.1002/nme.1296

Here, only pure interpolation is allowed. That is, attempting to evaluate the
interpolated function (or its derivatives) outside of the sampling domain
returns zero.

This particular interpolation method guarantees that the interpolated object
has continuous first derivatives, *mixed* second derivatives (i.e.,
d2f/dxdy, d2f/dxdz and d2f/dydz) in addition to d3f/dxdydz.

In *some* cases, the other second derivatives, that is, d2f/dx2, d2f/dy2
and d2f/dz2, may be continuous. The same applies to the third derivatives.
This depends strongly on the smoothness of the *actual* function.
"""

# The cython library contains a lot of useful functionalities, such as
# compiler flags;
cimport cython

import numpy as np

cdef class TricubicNonperiodicInterpolator:
    """
    TricubicNonperiodicInterpolator(x, y, z, data, periodic)

    Constructor for a TricubicNonperiodicInterpolator object. Intended for use
    on a Cartesian, three-dimensional grid of rectangular parallelepipeds.
    The grid spacings need not be the same along any pair of axes.

    param: x    -- A 1D numpy array of np.float64, defining the coordinates
                   along the first axis, at which the function has been
                   sampled. Must be strictly increasing.
                   *IMPORTANT*: len(x) >= 4.
    param: y    -- A 1D numpy array of np.float64, defining the coordinates
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
    def __cinit__(self, double[::1] x not None, double[::1] y not None,
            double[::1] z not None, double[:,:,::1] data not None):
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

        self.x_min = x[0]
        self.y_min = y[0]
        self.z_min = z[0]
        self.x_max = x[x.shape[0]-1]
        self.y_max = y[y.shape[0]-1]
        self.z_max = z[z.shape[0]-1]
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0]
        self.nx = x.shape[0]
        self.ny = y.shape[0]
        self.nz = z.shape[0]
        self.data = data[:self.nx-1,:self.ny-1,:self.nz-1]
        self.calibrated = 0
        # Explicitly set each element of the matrix A, using the predefined
        # 64-by-64 matrix in the helper file coeff_.h
        for j in range(64):
            for i in range(64):
                self.A[i][j] = get_coeff(&i,&j)
        # Precompute the necessary derivatives in order to avoid
        # awkward and convoluted computations of derivatives near edges,
        # by making use of NumPy's gradient function;
        # The 'derivatives' are computed using unitary grid spacings,
        # to be rescaled with the real grid spacings at a later point
        # (for algorithmic simplicity)
        self.data_derx = np.gradient(data,1,1,1,edge_order=2)[0]
        self.data_dery = np.gradient(data,1,1,1,edge_order=2)[1]
        self.data_derz = np.gradient(data,1,1,1,edge_order=2)[2]
        self.data_derxy = np.gradient(self.data_derx,1,1,1,edge_order=2)[1]
        self.data_derxz = np.gradient(self.data_derx,1,1,1,edge_order=2)[2]
        self.data_deryz = np.gradient(self.data_dery,1,1,1,edge_order=2)[2]
        self.data_derxyz = np.gradient(self.data_derxy,1,1,1,edge_order=2)[2]

    def ev(self, double x, double y, double z,
            int kx = 0, int ky = 0, int kz = 0):
        # A custom, thin Python wrapper for the _ev_ function, defined at C
        # level
        """
        TricubicNonperiodicInterpolator.ev(x, y, z, kx, ky, kz)

        Evaluate the interpolated function, or its derivatives,  at a single
        point. Attempts at evaluation outside of the sampling domain yield
        zero.

        param: x   -- Double-precision coordinate along the x axis.
        param: y   -- Double-precision coordinate along the y axis.
        param: z   -- Double-precision coordinate along the z axis.
        OPTIONAL:
        param: kx  -- Integer specifying the order of the partial derivative
                      along the x axis. 0 <= kx <= 3. DEFAULT: kx = 0.
        param: ky  -- Integer specifying the order of the partial derivative
                      along the y axis. 0 <= ky <= 3. DEFAULT: ky = 0.
        param: kz  -- Integer specifying the order of the partial derivative
                      along the z axis. 0 <= kz <= 3. DEFAULT: kz = 0.

        return:       Double-precision interpolated value.
        """
        return self._ev_(x, y, z, kx, ky, kz)

    # The following Cython compilation flags turn off the Pythonic
    # bounds check and wraparound functionality with regards to
    # array indexing, in addition to zero division / modulo zero safeguards.
    #
    # This is done for reasons of efficiency. Use with caution.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _ev_(self, double x, double y, double z, int kx, int ky, int kz):
        # The C-level function which evaluates the interpolated function (or
        # its derivatives) in a single point.

        # Local variables:
        cdef:
            double res = 0.             # Zero-initalizing the return variable
            int x_ind, y_ind, z_ind     # Indices defining the reference
                                        # corner within the interpolation voxel
            int i, j, k, w              # Loop counters
            double cont                 # Temporary variable, needed in order
                                        # to properly compute the interpolated
                                        # function value

        # Derivatives of negative orders are not well-defined:
        if(kz < 0 or ky < 0 or kz < 0):
            raise RuntimeError("Derivative order must be nonnegative.")

        # Seeing as this is a *cubic* interpolator, taking derivatives of higher
        # order than 3 will return zero anyways:
        if(kx > 3 or ky > 3 or kz > 3):
            raise RuntimeError("Derivative order can't be larger than 3.")

        # Return zero if one attempts to evaluate the interpolant outside of
        # the sampling domain
        if (x < self.x_min or x > self.x_max
                or y < self.y_min or y > self.y_max
                or z < self.z_min or z > self.z_max):
                return 0

        # Determine the relative coordinates of the point in question,
        # within its voxel
        x = (x-self.x_min)/self.dx
        y = (y-self.y_min)/self.dy
        z = (z-self.z_min)/self.dz

        # Find indices of voxel reference corner
        x_ind = int(c_floor(x))
        y_ind = int(c_floor(y))
        z_ind = int(c_floor(z))

        # If we are at a domain boundary, take a step backwards into
        # the interpolation parallelepiped.
        if x_ind == self.nx - 1:
            x_ind -= 1
        if y_ind == self.ny - 1:
            y_ind -= 1
        if z_ind == self.nz - 1:
            z_ind -= 1

        # Find relative position within voxel
        x -= x_ind
        y -= y_ind
        z -= z_ind

        # If the previous interpolator evaluation was performed within the
        # same voxel as the one we're looking at now, we don't need to
        # recompute the interpolation coefficients:
        if(self.calibrated == 0
                or x_ind != self.xi or y_ind != self.yi or z_ind != self.zi):
            self._calibrate_(x_ind,y_ind,z_ind)

        # Loop over the required powers of the voxel coordinates
        for k in range(kz, 4):
            for j in range(ky, 4):
                for i in range(kx, 4):
                    cont = self.coeffs[self._ind_(i,j,k)]*c_pow(x,i-kx)\
                                            *c_pow(y,j-ky)*c_pow(z,k-kz)
                    # Explicitly handle prefactors from derivatives:
                    for w in range(kx):
                        cont *= (i-w)
                    for w in range(ky):
                        cont *= (j-w)
                    for w in range(kz):
                        cont *= (k-w)
                    res += cont

        # Because the derivatives, as computed in self._calibrate_periodic_
        # or self._calibrate_nonperiodic_, are not scaled with the grid spacings,
        # we must do so explicitly in order to obtain a properly scaled return
        # variable:
        return res/(c_pow(self.dx,kx)*c_pow(self.dy,ky)*c_pow(self.dz,kz))

    cdef int _ind_(self, int i, int j, int k):
        # A convenience function, used to transform (tuples of) integers to
        # a single index for the interpolation coefficient array
        return(i + 4*j + 16*k)


    def ev_grid(self, double[::1] x, double[::1] y, double[::1] z,
                    int kx = 0, int ky = 0, int kz = 0):
        # A custom, thin Python wrapper for the _ev_ function, defined at C
        # level
        """
        TricubicNonperiodicInterpolator.ev_grid(x, y, z, kx, ky, kz)

        Evaluate the interpolated function, or its derivatives, at the grid
        spanned by the input x, y and z arrays. Attempts at evaluation
        outside of the sampling domain yield zero.

        param: x   -- A 1D NumPy array of np.float64, containing the points along
                      the x abscissa at which an interpolated value is sought
        param: y   -- A 1D NumPy array of np.float64, containing the points along
                      the y abscissa at which an interpolated value is sought
        param: z   -- A NumPy array of np.float64, containing the points along
                      the z abscissa at which an interpolated value is sought
        OPTIONAL:
        param: kx  -- Integer specifying the order of the partial derivative
                      along the x axis. 0 <= kx <= 3. DEFAULT: kx = 0.
        param: ky  -- Integer specifying the order of the partial derivative
                      along the y axis. 0 <= ky <= 3. DEFAULT: ky = 0.
        param: kz  -- Integer specifying the order of the partial derivative
                      along the z axis. 0 <= kz <= 3. DEFAULT: kz = 0.

        return:       A NumPy array of np.float64 interpolated values.
                      Shape: (len(x),len(y),len(z)).
        """
        return self._ev_grid_(x, y, z, kx, ky, kz)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.ndarray[np.float64_t, ndim=3] _ev_grid_(self,double[::1] x,
                                                         double[::1] y,
                                                         double[::1] z,
                                                         int kx, int ky, int kz):
        cdef:
            int i, j, k
            int nx = x.shape[0]
            int ny = y.shape[0]
            int nz = z.shape[0]
            np.ndarray[np.float64_t, ndim=3] res = np.empty((x.shape[0],
                                                             y.shape[0],
                                                             z.shape[0]),
                                                             dtype=np.float64)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    res[i,j,k] = self._ev_(x[i], y[j], z[k], kx, ky, kz)

        return res


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _calibrate_(self, int ix, int iy, int iz):

        # Values of f(x,y,z) at the corners of the voxel
        self._set_vals_(ix, iy, iz)
        # First derivatives of f(x,y,z) at the corners of the voxel
        self._set_derivs_(ix, iy, iz)
        # Mixed second derivatives of f(x,y,z) at the corners of the voxel
        self._set_mxd_2derivs_(ix, iy, iz)
        ## Values of d3f/dxdydz at the corners of the voxel
        self._set_mxd_3deriv_(ix, iy, iz)
        # Convert voxel values and partial derivatives to interpolation
        # coefficients
        self._compute_coeffs_by_blas_dgemv_()
        # Remember the configuration for the next call
        self.xi, self.yi, self.zi = ix, iy, iz
        self.calibrated = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_vals_(self, int x, int y, int z):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data = self.data
        # Values of f(x,y,z) at the corners of the voxel
        psi[0]  = data[x,y,z]
        psi[1]  = data[x+1,y,z]
        psi[2]  = data[x,y+1,z]
        psi[3]  = data[x+1,y+1,z]
        psi[4]  = data[x,y,z+1]
        psi[5]  = data[x+1,y,z+1]
        psi[6]  = data[x,y+1,z+1]
        psi[7]  = data[x+1,y+1,z+1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_derivs_(self, int x, int y, int z):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data_derx = self.data_derx
            double[:,:,::1] data_dery = self.data_dery
            double[:,:,::1] data_derz = self.data_derz
        # Values of df/dx at the corners of the voxel
        psi[8]  = data_derx[x,y,z]
        psi[9]  = data_derx[x+1,y,z]
        psi[10] = data_derx[x,y+1,z]
        psi[11] = data_derx[x+1,y+1,z]
        psi[12] = data_derx[x,y,z+1]
        psi[13] = data_derx[x+1,y,z+1]
        psi[14] = data_derx[x,y+1,z+1]
        psi[15] = data_derx[x+1,y+1,z+1]
        # Values of df/dy at the corners of the voxel
        psi[16] = data_dery[x,y,z]
        psi[17] = data_dery[x+1,y,z]
        psi[18] = data_dery[x,y+1,z]
        psi[19] = data_dery[x+1,y+1,z]
        psi[20] = data_dery[x,y,z+1]
        psi[21] = data_dery[x+1,y,z+1]
        psi[22] = data_dery[x,y+1,z+1]
        psi[23] = data_dery[x+1,y+1,z+1]
        # Values of df/dz at the corners of the voxel
        psi[24] = data_derz[x,y,z]
        psi[25] = data_derz[x+1,y,z]
        psi[26] = data_derz[x,y+1,z]
        psi[27] = data_derz[x+1,y+1,z]
        psi[28] = data_derz[x,y,z+1]
        psi[29] = data_derz[x+1,y,z+1]
        psi[30] = data_derz[x,y+1,z+1]
        psi[31] = data_derz[x+1,y+1,z+1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_mxd_2derivs_(self, int x, int y, int z):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data_derxy = self.data_derxy
            double[:,:,::1] data_derxz = self.data_derxz
            double[:,:,::1] data_deryz = self.data_deryz
        # Values of d2f/dxdy at the corners of the voxel
        psi[32] = data_derxy[x,y,z]
        psi[33] = data_derxy[x+1,y,z]
        psi[34] = data_derxy[x,y+1,z]
        psi[35] = data_derxy[x+1,y+1,z]
        psi[36] = data_derxy[x,y,z+1]
        psi[37] = data_derxy[x+1,y,z+1]
        psi[38] = data_derxy[x,y+1,z+1]
        psi[39] = data_derxy[x+1,y+1,z+1]
        # Values of d2f/dxdz at the corners of the voxel
        psi[40] = data_derxz[x,y,z]
        psi[41] = data_derxz[x+1,y,z]
        psi[42] = data_derxz[x,y+1,z]
        psi[43] = data_derxz[x+1,y+1,z]
        psi[44] = data_derxz[x,y,z+1]
        psi[45] = data_derxz[x+1,y,z+1]
        psi[46] = data_derxz[x,y+1,z+1]
        psi[47] = data_derxz[x+1,y+1,z+1]
        # Values of d2f/dydz at the corners of the voxel
        psi[48] = data_deryz[x,y,z]
        psi[49] = data_deryz[x+1,y,z]
        psi[50] = data_deryz[x,y+1,z]
        psi[51] = data_deryz[x+1,y+1,z]
        psi[52] = data_deryz[x,y,z+1]
        psi[53] = data_deryz[x+1,y,z+1]
        psi[54] = data_deryz[x,y+1,z+1]
        psi[55] = data_deryz[x+1,y+1,z+1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_mxd_3deriv_(self, int x, int y, int z):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data_derxyz = self.data_derxyz
        # Values of d3f/dxdydz at the corners of the voxel
        psi[56] = data_derxyz[x,y,z]
        psi[57] = data_derxyz[x+1,y,z]
        psi[58] = data_derxyz[x,y+1,z]
        psi[59] = data_derxyz[x+1,y+1,z]
        psi[60] = data_derxyz[x,y,z+1]
        psi[61] = data_derxyz[x+1,y,z+1]
        psi[62] = data_derxyz[x,y+1,z+1]
        psi[63] = data_derxyz[x+1,y+1,z+1]

    cdef _compute_coeffs_by_blas_dgemv_(self):
        # Computes matrix-vector product needed to identify interpolation
        # coefficients within a given voxel.
        #
        # Does so by calling the BLAS level 2 function 'dgemv', included
        # from within the SciPy linear algebra library.
        #
        # Detailed documentation is available at e.g.
        #       http://www.netlib.org/lapack/explore-html/dc/da8/dgemv_8f.html
        #       (checked Feb. 7, 2018)
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
            char *trans = 'T'
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
