
import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport fmod as c_fmod, floor as c_floor, pow as c_pow

cdef class TrilinearInterpolator:
    """
    This class provides a Python object interface to optimized C code which
    enables trilinear interpolation in three dimensions, where periodic boundary
    conditions in all three dimensions is implicitly assumed.
    """
    cdef:
        # The minimum boundaries of the computational domain
        # (needed in order to properly transform input coordinates to
        # the relevant interpolation voxel)
        double x1_min, x2_min, x3_min
        # The number of points along each axis
        int n1, n2, n3
        # The grid spacings are needed for the finite difference approximation
        # of derivatives, in addition to transform input coordinates to
        # the relevant interpolation voxel
        double dx1, dx2, dx3
        # The 3D-data to be interpolated
        double[:,:,::1] data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # Turning off the explicit bounds check and wraparound functionality of
    # e.g. NumPy arrays locally, in order to improve efficiency.
    def __init__(self,double[::1] x1 not None,double[::1] x2 not None,
            double[::1] x3 not None,double[:,:,::1] data not None):
        """
        TrilinearInterpolator(x1, x2, x3, data)

        Constructor for a TrilinearInterpolator object. Intended for use on
        a Cartesian, three-dimensional grid with equidistant grid spacing.
        The grid spacings need not be the same along any pair of axes.

        param: x1   -- A 1D numpy array of np.float64, defining the coordinates
                       along the first axis, at which the function has been
                       sampled. *Important*: len(x1) > 1.
        param: x2   -- A 1D numpy array of np.float64, defining the coordinates
                       along the second axis, at which the function has been
                       sampled. Important: len(x2) > 1.
        param: x3   -- A 1D numpy array of np.float64, defining the coordinates
                       along the third axis, at which the function has been
                       sampled. Important: len(x3) > 1.
        param: data -- A 3D numpy array of np.float64, containing the sampled
                       function values on the grid spanned by x1, x2 and x3.
                       Shape: (len(x1),len(x2),len(x3)).
        """

        if(x1.shape[0] == 0 or x2.shape[0] == 0 or x3.shape[0] == 0):
            raise RuntimeError("Abscissa vectors must have a positive number of\
                                elements!")

        if(x1.shape[0] != data.shape[0] or x2.shape[0] != data.shape[1]
                or x3.shape[0] != data.shape[2]):
            raise RuntimeError("Input data not properly aligned. See\
                                constructor docstring for details.")

        self.x1_min = x1[0]
        self.x2_min = x2[0]
        self.x3_min = x3[0]
        self.dx1 = x1[1]-x1[0]
        self.dx2 = x2[1]-x2[0]
        self.dx3 = x3[1]-x3[0]
        self.n1 = x1.shape[0]
        self.n2 = x2.shape[0]
        self.n3 = x3.shape[0]
        self.data = data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef double ev(self,double x1, double x2, double x3):
        cdef:
            double cubevals[8]
            double planevals[4]
            double linevals[2]
            double[:,:,::1] data = self.data
            int ix1,ix1p1,ix2,ix2p1,ix3,ix3p1
            double x1d,x2d,x3d
	# Determine the relative coordinates of the point in question, within
	# its voxel
        x1 = c_fmod((x1-self.x1_min)/self.dx1,self.n1)
        x2 = c_fmod((x2-self.x2_min)/self.dx2,self.n2)
        x3 = c_fmod((x3-self.x3_min)/self.dx3,self.n3)

	# Enforce periodic BC
        while(x1 < 0):
            x1 += self.n1
        while(x2 < 0):
            x2 += self.n2
        while(x3 < 0):
            x3 += self.n3

        # Find coordinates of voxel reference coordinates
        ix1 = int(c_floor(x1))
        ix2 = int(c_floor(x2))
        ix3 = int(c_floor(x3))

        # Find relative position within voxel
        # These coordinates are normalized, and thus perfect for use in
        # linear interpolation
        x1 -= ix1
        x2 -= ix2
        x3 -= ix3

        # Find indices of the remaining voxel corners by making use of the
        # innate periodicity
        ix1p1 = (ix1+1)%self.n1
        ix2p1 = (ix2+1)%self.n2
        ix3p1 = (ix3+1)%self.n3

        # Extract function values at voxel corners
        cubevals[0] = data[ix1,ix2,ix3]
        cubevals[1] = data[ix1p1,ix2,ix3]
        cubevals[2] = data[ix1,ix2p1,ix3]
        cubevals[3] = data[ix1p1,ix2p1,ix3]
        cubevals[4] = data[ix1,ix2,ix3p1]
        cubevals[5] = data[ix1p1,ix2,ix3p1]
        cubevals[6] = data[ix1,ix2p1,ix3p1]
        cubevals[7] = data[ix1p1,ix2p1,ix3p1]

        # Perform linear interpolation along the x1 axis, yielding a
        # rectangle in the (x2,x3) plane, to be interpolated further
        planevals[0] = cubevals[0]*(1-x1) + cubevals[1]*x1
        planevals[1] = cubevals[2]*(1-x1) + cubevals[3]*x1
        planevals[2] = cubevals[4]*(1-x1) + cubevals[5]*x1
        planevals[3] = cubevals[6]*(1-x1) + cubevals[7]*x1

        # Perform linear interpolation along the x2 axis, yielding two points
        # on the x3 axis, to be interpolated further
        linevals[0] = planevals[0]*(1-x2) + planevals[1]*x2
        linevals[1] = planevals[2]*(1-x2) + planevals[3]*x2

        # Perform linear interpolation along the x3 axis
        return linevals[0]*(1-x3) + linevals[1]*x3

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def ev_grid(self,double[:,:,::1] x1, double[:,:,::1] x2, double[:,:,::1] x3):
        cdef:
            int i, j, k
            int x1_sh0 = x1.shape[0], x1_sh1 = x1.shape[1], x1_sh2 = x1.shape[2]
            int x2_sh0 = x2.shape[0], x2_sh1 = x2.shape[1], x2_sh2 = x2.shape[2]
            int x3_sh0 = x3.shape[0], x3_sh1 = x3.shape[1], x3_sh2 = x3.shape[2]

        if(x1_sh0 != x2_sh0 or x1_sh0 != x3_sh0 or x2_sh0 != x3_sh0
                or x1_sh1 != x2_sh1 or x1_sh1 != x3_sh1 or x2_sh1 != x3_sh1
                or x1_sh2 != x2_sh2 or x1_sh2 != x3_sh2 or x2_sh2 != x3_sh2):
            raise RuntimeError("Array dimensions inconsistent!")

        cdef np.ndarray[np.float64_t,ndim=3] res = np.empty((x1_sh0,
                                                            x1_sh1,
                                                            x1_sh2),
                                                            dtype=np.float64)

        for k in range(x1_sh2):
            for j in range(x1_sh1):
                for i in range(x1_sh0):
                    res[i,j,k] = self.ev(x1[i,j,k],x2[i,j,k],x3[i,j,k])

        return res
