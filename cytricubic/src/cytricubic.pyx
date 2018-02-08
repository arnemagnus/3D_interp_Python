"""
This module contains an object oriented implementation of a local tricubic
polynomial interpolation routine in 3D, the theoretical foundation of which is
found in

    Lekien, F and Marsden, J (2005):
        'Tricubic Interpolation in Three Dimensions',
        in Journal of Numerical Methods and Engineering(63), pp. 455-471,
	doi:10.1002/nme.1296

Two interpolation modes are available:
    a) Periodic boundary conditions
    b) Pure interpolation, i.e., attempting to evaluate the interpolation
       object outside of the sampling domain returns zero

Unless otherwise specified, periodic boundary conditions are assumed.

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

# The other import commands are defined in the accompanying Cython header file:
cimport cytricubic

cdef class TricubicInterpolator:
    """
    TricubicInterpolator(x, y, z, data, periodic)

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
    OPTIONAL:
    param: periodic -- Boolean flag indicating whether or not to use
                       periodic boundary conditions. Default: True.
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
            double[::1] z not None, double[:,:,::1] data not None,
            bint periodic = 1):
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

        self.periodic = periodic
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
        # If periodic boundary conditions are used, we map the last index in
        # each respective direction to the corresponding first index.
        # Thus, there is no need to store the data corresponding to the *actual*
        # last indices in any direction:
        if self.periodic:
            self.nx -= 1
            self.ny -= 1
            self.nz -= 1
        # Explicitly specify the data limits in case of periodic boundary
        # conditions, cf. above.
        self.data = data[:self.nx-1,:self.ny-1,:self.nz-1]
        self.calibrated = 0
        # Explicitly set each element of the matrix A, using the predefined
        # 64-by-64 matrix in the helper file coeff_.h
        for j in range(64):
            for i in range(64):
                self.A[i][j] = get_coeff(&i,&j)

    def ev(self, double x, double y, double z,
            int kx = 0, int ky = 0, int kz = 0):
        # A custom, thin Python wrapper for the _ev_ function, defined at C
        # level
        """
        TricubicInterpolator.ev(x, y, z, kx, ky, kz)

        Evaluate the interpolated function, or its derivatives,  at a single
        point.

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

        if self.periodic:
            # Determine the relative coordinates of the point in question, within
            # the entire interpolation parallelepiped.
            #
            # In transforming the physical coordinates to voxel indices,
            # we subtract the minimum boundary values of the physical domain.
            # This is due to the interpolation parallelepiped being zero-indexed.
            # Then, we divide through by the grid spacings, in order to obtain
            # normalized voxel coordinates.
            # Lastly, we take the modulo by the number of grid points in order to
            # enforce periodic boundary conditions.
            x = c_fmod((x-self.x_min)/self.dx,self.nx)
            y = c_fmod((y-self.y_min)/self.dy,self.ny)
            z = c_fmod((z-self.z_min)/self.dz,self.nz)

	    # As a second step on the path of enforcing periodic boundary positions,
            # we ensure that the normalized coordinates lie within the intervals
            # )0,ni(, i = x, y or z, respectively.
            while(x < 0):
                x += self.nx
            while(y < 0):
                y += self.ny
            while(z < 0):
                z += self.nz

            # The integer part of the normalized coordinates define the
            # reference corner of the voxel within which we shall interpolate
            x_ind = int(c_floor(x))
            y_ind = int(c_floor(y))
            z_ind = int(c_floor(z))

            # The decimal part of the normalized coordinates are needed to
            # evaluate the voxel-local tricubic polynomial, hence:
            x -= x_ind
            y -= y_ind
            z -= z_ind

            # If the previous interpolator evaluation was performed within the
            # same voxel as the one we're looking at now, we don't need to
            # recompute the interpolation coefficients:
            if(self.calibrated == 0
                or x_ind != self.xi or y_ind != self.yi or z_ind != self.zi):
                self._calibrate_periodic_(x_ind,y_ind,z_ind)
        else:
            if (x < self.x_min or x > self.x_max
                    or y < self.y_min or y > self.y_max
                    or z < self.z_min or z > self.z_max):
                return 0
            else:
                return -42 # Placeholder until the logic is in place.

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
        # or self._calibrate_nonperiodic_ are not scaled with the grid spacings,
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
        TricubicInterpolator.ev_grid(x, y, z, kx, ky, kz)

        Evaluate the interpolated function, or its derivatives, at the grid
        spanned by the input x, y and z arrays.

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

        cdef np.ndarray[np.float64_t, ndim=3] res = np.empty((x.shape[0],
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

    cdef _set_periodic_voxel_indices_(self,
                                     int ix, int *ixm1, int *ixp1, int *ixp2,
                                     int iy, int *iym1, int *iyp1, int *iyp2,
                                     int iz, int *izm1, int *izp1, int *izp2):
        ixm1[0] = (ix-1)%self.nx
        ixp1[0] = (ix+1)%self.nx
        ixp2[0] = (ix+2)%self.nx

        iym1[0] = (iy-1)%self.ny
        iyp1[0] = (iy+1)%self.ny
        iyp2[0] = (iy+2)%self.ny

        izm1[0] = (iz-1)%self.nz
        izp1[0] = (iz+1)%self.nz
        izp2[0] = (iz+2)%self.nz

        if(ixm1[0] < 0):
            ixm1[0] += self.nx
        if(ixp1[0] < 0):
            ixp1[0] += self.nx
        if(ixp2[0] < 0):
            ixp2[0] += self.nx

        if(iym1[0] < 0):
            iym1[0] += self.ny
        if(iyp1[0] < 0):
            iyp1[0] += self.ny
        if(iyp2[0] < 0):
            iyp2[0] += self.ny

        if(izm1[0] < 0):
            izm1[0] += self.nz
        if(izp1[0] < 0):
            izp1[0] += self.nz
        if(izp2[0] < 0):
            izp2[0] += self.nz

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _calibrate_periodic_(self, int ix, int iy, int iz):
        cdef:
            int ixm1, ixp1, ixp2
            int iym1, iyp1, iyp2
            int izm1, izp1, izp2

        # Precompute the voxel indices plus/minus one and two, in order to
        # explicitly provide the Python array wraparound and bounds check
        # feature, so that we can disable it at a Python level (to enhance
        # performance)
        self._set_periodic_voxel_indices_(ix, &ixm1, &ixp1, &ixp2,
                                          iy, &iym1, &iyp1, &iyp2,
                                          iz, &izm1, &izp1, &izp2)

        # Values of f(x,y,z) at the corners of the voxel
        self._set_periodic_vals_(ix, ixp1, iy, iyp1, iz, izp1)
        # First derivatives of f(x,y,z) at the corners of the voxel
        self._set_periodic_derivs_(ix, ixm1, ixp1, ixp2,
                                   iy, iym1, iyp1, iyp2,
                                   iz, izm1, izp1, izp2)
        # Mixed second derivatives of f(x,y,z) at the corners of the voxel
        self._set_periodic_mxd_2derivs_(ix, ixm1, ixp1, ixp2,
                                        iy, iym1, iyp1, iyp2,
                                        iz, izm1, izp1, izp2)
        # Values of d3f/dxdydz at the corners of the voxel
        self._set_periodic_mxd_3deriv_(ix, ixm1, ixp1, ixp2,
                                       iy, iym1, iyp1, iyp2,
                                       iz, izm1, izp1, izp2)
        # Convert voxel values and partial derivatives to interpolation
        # coefficients

        self._compute_coeffs_by_blas_dgemv_()
        # Remember the configuration for the next call
        self.xi, self.yi, self.zi = ix, iy, iz
        self.calibrated = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_periodic_vals_(self, int x, int xp1, int y, int yp1, int z, int zp1):
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
    cdef _set_periodic_derivs_(self, int x, int xm1, int xp1, int xp2,
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
    cdef _set_periodic_mxd_2derivs_(self, int x, int xm1, int xp1, int xp2,
                                          int y, int ym1, int yp1, int yp2,
                                          int z, int zm1, int zp1, int zp2):
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
    cdef _set_periodic_mxd_3deriv_(self,int x, int xm1, int xp1, int xp2,
                                        int y, int ym1, int yp1, int yp2,
                                        int z, int zm1, int zp1, int zp2):
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

    cpdef _compute_coeffs_by_blas_dgemv_(self):
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
