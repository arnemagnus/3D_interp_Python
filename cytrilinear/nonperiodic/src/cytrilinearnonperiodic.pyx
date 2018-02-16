"""
This module contains an object oriented implementation of a trilinear
interpolation routine routine in 3D, allowing only pure interpolation. That is,
attempting to evaluate the interpolated function outside of the sampling
domain returns zero.
"""

# The Cython library contains a lot of useful functionalities, such as compiler
# flags;
cimport cython

import numpy as np

cdef class TrilinearNonperiodicInterpolator:
    """
    TrilinearNonperidocInterpolator(x, y, z, data, periodic)

    Constructor for a TrilinearNonperiodicInterpolator object. Intended for use
    on a Cartesian, three-dimensional grid with equidistant grid spacing.
    The grid spacings need not be the same along any pair of axes.

    param: x    -- A 1D numpy array of np.float64, defining the coordinates
                   along the first axis, at which the function has been
                   sampled. Must be strictly increasing.
                   *IMPORTANT*: len(x) > 1.
    param: y    -- A 1D numpy array of np.float64, defining the coordinates
                   along the second axis, at which the function has been
                   sampled. Must be strictly increasing.
                   *IMPORTANT*: len(y) > 1.
    param: z    -- A 1D numpy array of np.float64, defining the coordinates
                   along the third axis, at which the function has been
                   sampled. Must be strictly increasing.
                   *IMPORTANT*: len(z) > 1.
    param: data -- A 3D numpy array of np.float64, containing the sampled
                   function values on the grid spanned by x, y and z.
                   Shape: (len(x),len(y),len(z)).
    """

    # Turning off the explicit bounds check and wraparound functionality of
    # e.g. NumPy arrays locally, in order to improve efficiency.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self,double[::1] x not None,double[::1] y not None,
            double[::1] z not None,double[:,:,::1] data not None):

        if(x.shape[0] == 0 or y.shape[0] == 0 or z.shape[0] == 0):
            raise RuntimeError("Abscissa vectors must have a positive number of\
                                elements!")

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

    def ev(self, double x, double y, double z):
        # A custom, thin Python wrapper for the _ev_ function, defined at
        # C level
        """
        TrilinearNonperiodicInterpolator.ev(x, y, z)

        Evaluate the interpolated function at a single point (x, y, z).
        Evaluation outside of the sampling domain will return zero.

        param: x -- Double-precision coordinate along the x axis
        param: y -- Double-precision coordinate along the y axis
        param: z -- Double-precision coordinate along the z axis

        return:     Double-precision interpolated value.
        """
        return self._ev_(x, y, z)

    # The following Cython compilation flags turn off the Pythonic bounds check
    # and wraparound functionalities with regards to array indexing, in
    # addition to a lower level check that any typed memoryviews are
    # initialized.
    #
    # These safeguards are generally nice to have, but drastically decrease
    # performance. So, by being very explicit regarding indexing, we can turn
    # them off - at the risk of segfaults (best case scenario) or corrupted data
    # (worst case scenario), should we make an error.
    #
    # In short: These should be used with caution.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _ev_(self, double x, double y, double z):
        # A C level function which evaluates the interpolated function in a
        # single point.

        # Local variables:
        cdef:
            double cubevals[8]                  # Container for the function
                                                # values at the voxel corners
            double planevals[4]                 # Container for the function
                                                # values at the rectangle
                                                # corners, resulting from
                                                # interpolating the voxel
                                                # along one axis
            double linevals[2]                  # Container for the function
                                                # values at the line segment,
                                                # resulting from interpolating
                                                # the rectangle along one axis
            int ix, ixp1, iy, iyp1, iz, izp1    # Indices defining the corners
                                                # of the interpolation voxel
                                                # of interest

        if (x < self.x_min or x > self.x_max
                or y < self.y_min or y > self.y_max
                or z < self.z_min or z > self.z_max):
            return 0

        self._set_voxel_indices_(&x, &y, &z, &ix, &ixp1, &iy, &iyp1, &iz, &izp1)

        # Extract function values at voxel corners
        cubevals[0] = self.data[  ix,   iy,  iz]
        cubevals[1] = self.data[ixp1,   iy,  iz]
        cubevals[2] = self.data[  ix, iyp1,  iz]
        cubevals[3] = self.data[ixp1, iyp1,  iz]
        cubevals[4] = self.data[  ix,   iy,izp1]
        cubevals[5] = self.data[ixp1,   iy,izp1]
        cubevals[6] = self.data[  ix, iyp1,izp1]
        cubevals[7] = self.data[ixp1, iyp1,izp1]

        # In either of the two above function calls to set the voxel indices,
        # the physical x, y and z coordinates are converted to normalized
        # relative coordinates within the voxel with corners defined by
        # the indices ix, ixp1, iy, iyp1, iz and izp1.
        #
        # The normalized relative coordinates are exactly what is needed
        # to perform proper linear interpolation, hence:

        # Perform linear interpolation along the x axis, yielding a
        # rectangle in the (y,z) plane, to be interpolated further
        planevals[0] = cubevals[0]*(1-x) + cubevals[1]*x
        planevals[1] = cubevals[2]*(1-x) + cubevals[3]*x
        planevals[2] = cubevals[4]*(1-x) + cubevals[5]*x
        planevals[3] = cubevals[6]*(1-x) + cubevals[7]*x

        # Perform linear interpolation along the y axis, yielding two points
        # on the z axis, to be interpolated further
        linevals[0] = planevals[0]*(1-y) + planevals[1]*y
        linevals[1] = planevals[2]*(1-y) + planevals[3]*y

        # Perform linear interpolation along the z axis
        return linevals[0]*(1-z) + linevals[1]*z


    @cython.cdivision(True)
    cdef _set_voxel_indices_(self, double* x, double *y, double *z,
                                               int *ix, int *ixp1,
                                               int *iy, int *iyp1,
                                               int *iz, int *izp1):
        # A C level function which computes the indices of the corners of the
        # interpolation voxel of interest, as well as the normalized relative
        # coordinates within said voxel, subject to pure interpolation mode
        # (that is, evaluation outside of the sampling domain returns zero).

        # Determine the relative coordinates of the point in question, within
        # its voxel
        x[0] = (x[0]-self.x_min)/self.dx
        y[0] = (y[0]-self.y_min)/self.dy
        z[0] = (z[0]-self.z_min)/self.dz

        # Find coordinates of voxel reference corner
        ix[0] = int(c_floor(x[0]))
        iy[0] = int(c_floor(y[0]))
        iz[0] = int(c_floor(z[0]))

        # If we are at a domain boundary, take a step backwards into the
        # the interpolation cube.
        #
        # This is done for simplicity, meaning one can perform linear
        # interpolation within a parallelepiped rather than handling the
        # boundary cases explicitly.
        if ix[0] == self.nx - 1:
            ix[0] -= 1
        if iy[0] == self.ny - 1:
            iy[0] -= 1
        if iz[0] == self.nz - 1:
            iz[0] -= 1

        # Find relative position within voxel
        x[0] -= ix[0]
        y[0] -= iy[0]
        z[0] -= iz[0]

        # Find indices of the remaining voxel corners
        ixp1[0] = ix[0] + 1
        iyp1[0] = iy[0] + 1
        izp1[0] = iz[0] + 1

    def ev_grid(self, double[::1] x, double[::1] y, double[::1] z):
        # A custom, thin Python wrapper for the _ev_grid_ function, defined at
        # C level
        """
        TrilinearNonperidocInterpolator.ev_grid(x, y, z)

        Evaluate the interpolated function at the grid spanned by the input
        x, y and z arrays..

        Evaluating the interpolated function outside of the sampling domain
        will return zero.

        param: x -- A 1D NumPy array of np.float64, containing the points along
                    the x abscissa at which an interpolated value is sought
        param: y -- A 1D NumPy array of np.float64, containing the points along
                    the y abscissa at which an interpolated value is sought
        param: z -- A 1D NumPy array of np.float64, containing the points along
                    the z abscissa at which an interpolated value is sought

        return:     A NumPy array of np.float64 interpolated values.
                    Shape: (len(x),len(y),len(z)).
        """
        return self._ev_grid_(x, y, z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.ndarray[np.float64_t,ndim=3] _ev_grid_(self,double[::1] x,
                                                        double[::1] y,
                                                        double[::1] z):
        # A C level function which evaluates the interpolated function on the
        # grid spanned by the input arrays x, y and z

        # Local variables:
        cdef:
            # Loop counters
            int i, j, k
            # Array sizes (constituent parts of the shape tuple of the generated
            # numpy array)
            int x_sz = x.shape[0], y_sz = y.shape[0], z_sz = z.shape[0]
            # The return variable:
            np.ndarray[np.float64_t,ndim=3] res = np.empty((x_sz,
                                                            y_sz,
                                                            z_sz),
                                                            dtype=np.float64)

        # Loop over all indices and fill in the return array one point at a time
        for k in range(z_sz):
            for j in range(y_sz):
                for i in range(x_sz):
                    res[i,j,k] = self._ev_(x[i],y[j],z[k])

        return res
