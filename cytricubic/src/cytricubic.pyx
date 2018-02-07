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

import numpy as np
cimport numpy as np

cimport cython

from scipy.linalg.blas import dgemv as sc_dgemv

from libc.math cimport fmod as c_fmod, floor as c_floor, pow as c_pow

# The attached header file "coeff_.h" contains the int (equiv. to
# numpy.int32) representation of the 64-by-64 matrix defining the linear system
# used for the three-dimensional interpolation.
#
# A simple wrapper function, aptly named "get_coeff", extracts the individual
# matrix elements, which is needed in order to set the system matrix elements
# for an instance of the TricubicInterpolator class. Explicitly typing
# the 64-times-64 coefficients by hand is out of the question.
cdef extern from "../include/coeff_.h":
    int get_coeff(int*, int*)


## Import matrix-vector product routine from BLAS level 2:
#cdef extern from "cblas.h":
#    ctypedef enum CBLAS_ORDER:
#        CblasRowMajor
#        CblasColumnMajor
#    ctypedef enum CBLAS_TRANSPOSE:
#        CblasNoTrans
#        CblasTrans
#        CblasConjTrans
#    void dgemv "cblas_dgemv"(CBLAS_ORDER order,
#                            CBLAS_TRANSPOSE transpose,
#                            int M, int N,
#                            double alpha, double *A, int lda,
#                            double *X, int incX,
#                            double beta, double *Y, int incY) nogil



cdef class TricubicInterpolator:
    """
    This class provides a Python object interface to optimized C code which
    enables tricubic interpolation in three dimensions, where periodic boundary
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
        # The 64-by-64 matrix which defines the linear 3D interpolation system
        double B[64][64]
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
        int i1, i2, i3

    cdef _solve_by_blas_dgemv_(self):
        cdef:
            int i, j
        # This function computes the matrix product which returns the
        # interpolation coefficients within a given voxel, making
        # use of the (C)BLAS routine dgemv.
        #
        # Detailed documentation of the
        self.coeffs = sc_dgemv(1.,self.B,self.psi)
#        for i in range(64):
#            self.coeffs[i] = 0.
#            for j in range(64):
#                self.coeffs[i] += self.B[i][j]*self.psi[j]


#        dgemv(CblasRowMajor,CblasNoTrans,64,64,
#                1.,&self.B[0][0],64,self.psi,1,0.,self.coeffs,1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # Turning off the explicit bounds check and wraparound functionality of
    # e.g. NumPy arrays locally, in order to improve efficiency.
    def __cinit__(self,double[::1] x1 not None,double[::1] x2 not None,
            double[::1] x3 not None,double[:,:,::1] data not None):
        """
        TricubicInterpolator(x1, x2, x3, data)

        Constructor for a TricubicInterpolator object. Intended for use on
        a Cartesian, three-dimensional grid of rectangular parallelepipeds.
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
        cdef:
            int i, j # Loop counters

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
        self.n1 = x1.shape[0]-1
        self.n2 = x2.shape[0]-1
        self.n3 = x3.shape[0]-1
        self.data = data[:self.n1-1,:self.n2-1,:self.n3-1]

        # Explicitly set each element of the matrix B, using the predefined
        # 64-by-64 matrix in the helper file coeff_.h
        for j in range(64):
            for i in range(64):
                self.B[i][j] = get_coeff(&i,&j)


        self.calibrated = 0

    def ev(self,double x1, double x2, double x3,
            int kx1 = 0, int kx2 = 0, int kx3 = 0):
        return self._ev_(x1,x2,x3,kx1,kx2,kx3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _ev_(self,double x1, double x2, double x3,
                int kx1, int kx2, int kx3):
        cdef:
            double dx1, dx2, dx3, dx3pow = 1., dx2pow = 1., res = 0.
            int x1_ind, x2_ind, x3_ind
            int i, j, k, ijk = 0
            double *coeffs = self.coeffs

        if(kx1 < 0 or kx2 < 0 or kx3 < 0):
            raise RuntimeError("Derivative order must be nonnegative.")

        if(kx1 > 3 or kx2 > 3 or kx3 > 3):
            raise RuntimeError("Derivative order can't be larger than 3.")

	# Determine the relative coordinates of the point in question, within
	# its voxel
        dx1 = c_fmod((x1-self.x1_min)/self.dx1,self.n1)
        dx2 = c_fmod((x2-self.x2_min)/self.dx2,self.n2)
        dx3 = c_fmod((x3-self.x3_min)/self.dx3,self.n3)

	# Enforce periodic BC
        while(dx1 < 0):
            dx1 += self.n1
        while(dx2 < 0):
            dx2 += self.n2
        while(dx3 < 0):
            dx3 += self.n3


        x1_ind = int(c_floor(dx1))
        x2_ind = int(c_floor(dx2))
        x3_ind = int(c_floor(dx3))

        if(self.calibrated == 0 or x1_ind != self.i1 or x2_ind != self.i2 or x3_ind != self.i3):
            self._calibrate_(x1_ind,x2_ind,x3_ind)

        dx1 -= x1_ind
        dx2 -= x2_ind
        dx3 -= x3_ind

        cdef:
            double cont
            int w

        for k in range(kx3,4):
            for j in range(kx2,4):
                for i in range(kx1,4):
                    cont = coeffs[self.ijk2n(i,j,k)]*c_pow(dx1,i-kx1)*c_pow(dx2,j-kx2)*c_pow(dx3,k-kx3)
                    for w in range(kx1):
                        cont *= (i-w)
                    for w in range(kx2):
                        cont *= (j-w)
                    for w in range(kx3):
                        cont *= (k-w)
                    res += cont
        return res/(c_pow(self.dx1,kx1)*c_pow(self.dx2,kx2)*c_pow(self.dx3,kx3))


    cdef int ijk2n(self,int i, int j, int k):
        return(i+4*j+16*k)


    def ev_grid(self,double[:,:,::1] x1, double[:,:,::1] x2, double[:,:,::1] x3,
                    int kx1 = 0, int kx2 = 0, int kx3 = 0):
        return self._ev_grid_(x1,x2,x3,kx1,kx2,kx3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _ev_grid_(self,double[:,:,::1] x1, double[:,:,::1] x2, double[:,:,::1] x3,
                int kx1, int kx2, int kx3):
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
                    res[i,j,k] = self._ev_(x1[i,j,k],x2[i,j,k],x3[i,j,k],
                                            kx1,kx2,kx3)

        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _calibrate_(self,int x1, int x2, int x3):
        cdef:
            int x1m1, x1p1, x1p2
            int x2m1, x2p1, x2p2
            int x3m1, x3p1, x3p2

        # Precompute the voxel indices plus/minus one and two, in order to
        # explicitly provide the Python array wraparound and bounds check
        # feature, so that we can disable it at a Python level (to enhance
        # performance)
        x1m1 = (x1-1)%self.n1
        x1p1 = (x1+1)%self.n1
        x1p2 = (x1+2)%self.n1

        x2m1 = (x2-1)%self.n2
        x2p1 = (x2+1)%self.n2
        x2p2 = (x2+2)%self.n2

        x3m1 = (x3-1)%self.n3
        x3p1 = (x3+1)%self.n3
        x3p2 = (x3+2)%self.n3

        if(x1m1 < 0):
            x1m1 += self.n1
        if(x1p1 < 0):
            x1p1 += self.n1
        if(x1p2 < 0):
            x1p2 += self.n1

        if(x2m1 < 0):
            x2m1 += self.n2
        if(x2p1 < 0):
            x2p1 += self.n2
        if(x2p2 < 0):
            x2p2 += self.n2

        if(x3m1 < 0):
            x3m1 += self.n3
        if(x3p1 < 0):
            x3p1 += self.n3
        if(x3p2 < 0):
            x3p2 += self.n3

        # Values of f(x1,x2,x3) at the corners of the voxel
        self._set_values_(x1,x1p1,x2,x2p1,x3,x3p1)
        # First derivatives of f(x1,x2,x3) at the corners of the voxel
        self._set_first_derivatives_(x1,x1m1,x1p1,x1p2,
                                        x2,x2m1,x2p1,x2p2,
                                        x3,x3m1,x3p1,x3p2)
        # Mixed second derivatives of f(x1,x2,x3) at the corners of the voxel
        self._set_second_drvtvs_(x1,x1m1,x1p1,x1p2,
                                 x2,x2m1,x2p1,x2p2,
                                 x3,x3m1,x3p1,x3p2)
        # Values of d3f/dxdydz at the corners of the voxel
        self._set_third_drvtv_(x1,x1m1,x1p1,x1p2,
                               x2,x2m1,x2p1,x2p2,
                               x3,x3m1,x3p1,x3p2)
        # Convert voxel values and partial derivatives to interpolation
        # coefficients
        self._solve_by_blas_dgemv_()
        # Remember the configuration for the next call
        self.i1, self.i2, self.i3 = x1, x2, x3
        self.calibrated = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_values_(self,int x1,int x1p1,int x2,int x2p1,int x3,int x3p1):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data = self.data
        # Values of f(x1,x2,x3) at the corners of the voxel
        psi[0]  = data[x1,x2,x3]
        psi[1]  = data[x1p1,x2,x3]
        psi[2]  = data[x1,x2p1,x3]
        psi[3]  = data[x1p1,x2p1,x3]
        psi[4]  = data[x1,x2,x3p1]
        psi[5]  = data[x1p1,x2,x3p1]
        psi[6]  = data[x1,x2p1,x3p1]
        psi[7]  = data[x1p1,x2p1,x3p1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_first_derivatives_(self,int x1,int x1m1,int x1p1,int x1p2,
                                        int x2, int x2m1, int x2p1, int x2p2,
                                        int x3, int x3m1, int x3p1, int x3p2):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data = self.data
        # Values of df/dx at the corners of the voxel
        psi[8]  = 0.5*(data[x1p1,x2,x3]-data[x1m1,x2,x3])
        psi[9]  = 0.5*(data[x1p2,x2,x3]-data[x1,x2,x3])
        psi[10] = 0.5*(data[x1p1,x2p1,x3]-data[x1m1,x2p1,x3])
        psi[11] = 0.5*(data[x1p2,x2p1,x3]-data[x1,x2p1,x3])
        psi[12] = 0.5*(data[x1p1,x2,x3p1]-data[x1m1,x2,x3p1])
        psi[13] = 0.5*(data[x1p2,x2,x3p1]-data[x1,x2,x3p1])
        psi[14] = 0.5*(data[x1p1,x2p1,x3p1]-data[x1m1,x2p1,x3p1])
        psi[15] = 0.5*(data[x1p2,x2p1,x3p1]-data[x1,x2p1,x3p1])
        # Values of df/dy at the corners of the voxel
        psi[16] = 0.5*(data[x1,x2p1,x3]-data[x1,x2m1,x3])
        psi[17] = 0.5*(data[x1p1,x2p1,x3]-data[x1p1,x2m1,x3])
        psi[18] = 0.5*(data[x1,x2p2,x3]-data[x1,x2,x3])
        psi[19] = 0.5*(data[x1p1,x2p2,x3]-data[x1p1,x2,x3])
        psi[20] = 0.5*(data[x1,x2p1,x3p1]-data[x1,x2m1,x3p1])
        psi[21] = 0.5*(data[x1p1,x2p1,x3p1]-data[x1p1,x2m1,x3p1])
        psi[22] = 0.5*(data[x1,x2p2,x3p1]-data[x1,x2,x3p1])
        psi[23] = 0.5*(data[x1p1,x2p2,x3p1]-data[x1p1,x2,x3p1])
        # Values of df/dz at the corners of the voxel
        psi[24] = 0.5*(data[x1,x2,x3p1]-data[x1,x2,x3m1])
        psi[25] = 0.5*(data[x1p1,x2,x3p1]-data[x1p1,x2,x3m1])
        psi[26] = 0.5*(data[x1,x2p1,x3p1]-data[x1,x2p1,x3m1])
        psi[27] = 0.5*(data[x1p1,x2p1,x3p1]-data[x1p1,x2p1,x3m1])
        psi[28] = 0.5*(data[x1,x2,x3p2]-data[x1,x2,x3])
        psi[29] = 0.5*(data[x1p1,x2,x3p2]-data[x1p1,x2,x3])
        psi[30] = 0.5*(data[x1,x2p1,x3p2]-data[x1,x2p1,x3])
        psi[31] = 0.5*(data[x1p1,x2p1,x3p2]-data[x1p1,x2p1,x3])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_second_drvtvs_(self,int x1,int x1m1,int x1p1,int x1p2,
                                   int x2,int x2m1,int x2p1,int x2p2,
                                   int x3,int x3m1,int x3p1,int x3p2):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data = self.data
        # Values of d2f/dxdy at the corners of the voxel
        psi[32] = 0.25*((data[x1p1,x2p1,x3]-data[x1m1,x2p1,x3])
                            - (data[x1p1,x2m1,x3]-data[x1m1,x2m1,x3]))
        psi[33] = 0.25*((data[x1p2,x2p1,x3]-data[x1,x2p1,x3])
                            -(data[x1p2,x2m1,x3]-data[x1,x2m1,x3]))
        psi[34] = 0.25*((data[x1p1,x2p2,x3]-data[x1m1,x2,x3])
                            -(data[x1p1,x2,x3]-data[x1m1,x2,x3]))
        psi[35] = 0.25*((data[x1p2,x2p2,x3]-data[x1,x2p2,x3])
                            -(data[x1p2,x2,x3]-data[x1,x2,x3]))
        psi[36] = 0.25*((data[x1p1,x2p1,x3p1]-data[x1m1,x2p1,x3p1])
                            -(data[x1p1,x2m1,x3p1]-data[x1m1,x2m1,x3p1]))
        psi[37] = 0.25*((data[x1p2,x2p1,x3p1]-data[x1,x2p1,x3p1])
                            -(data[x1p2,x2m1,x3p1]-data[x1,x2m1,x3p1]))
        psi[38] = 0.25*((data[x1p1,x2p2,x3p1]-data[x1m1,x2p2,x3p1])
                            -(data[x1p1,x2,x3p1]-data[x1m1,x2,x3p1]))
        psi[39] = 0.25*((data[x1p2,x2p2,x3p1]-data[x1,x2p2,x3p1])
                            -(data[x1p2,x2,x3p1]-data[x1,x2,x3p1]))
        # Values of d2f/dxdz at the corners of the voxel
        psi[40] = 0.25*((data[x1p1,x2,x3p1]-data[x1m1,x2,x3p1])
                            -(data[x1p1,x2,x3m1]-data[x1m1,x2,x3m1]))
        psi[41] = 0.25*((data[x1p2,x2,x3p1]-data[x1,x2,x3p1])
                            -(data[x1p2,x2,x3m1]-data[x1,x2,x3m1]))
        psi[42] = 0.25*((data[x1p1,x2p1,x3p1]-data[x1m1,x2p1,x3p1])
                            -(data[x1p1,x2p1,x3m1]-data[x1m1,x2p1,x3m1]))
        psi[43] = 0.25*((data[x1p2,x2p1,x3p1]-data[x1,x2p1,x3p1])
                            -(data[x1p2,x2p1,x3m1]-data[x1,x2p1,x3m1]))
        psi[44] = 0.25*((data[x1p1,x2,x3p2])-data[x1m1,x2,x3p2]
                            -(data[x1p1,x2,x3]-data[x1m1,x2,x3]))
        psi[45] = 0.25*((data[x1p2,x2,x3p2]-data[x1,x2,x3p2])
                            -(data[x1p2,x2,x3]-data[x1,x2,x3]))
        psi[46] = 0.25*((data[x1p1,x2p2,x3p2]-data[x1m1,x2p2,x3p2])
                            -(data[x1p1,x2p2,x3]-data[x1m1,x2p2,x3]))
        psi[47] = 0.25*((data[x1p2,x2p2,x3p2]-data[x1,x2p2,x3p2])
                            -(data[x1p2,x2p2,x3]-data[x1,x2p2,x3]))
        # Values of d2f/dydz at the corners of the voxel
        psi[48] = 0.25*((data[x1,x2p1,x3p1]-data[x1,x2m1,x3p1])
                            -(data[x1,x2p1,x3m1]-data[x1,x2m1,x3m1]))
        psi[49] = 0.25*((data[x1p1,x2p1,x3p1]-data[x1p1,x2m1,x3p1])
                            -(data[x1p1,x2p1,x3m1]-data[x1p1,x2m1,x3m1]))
        psi[50] = 0.25*((data[x1,x2p1,x3p1]-data[x1,x2m1,x3p1])
                            -(data[x1,x2p1,x3m1]-data[x1,x2m1,x3m1]))
        psi[51] = 0.25*((data[x1p1,x2p2,x3p1]-data[x1p1,x2,x3p1])
                            -(data[x1p1,x2p2,x3m1]-data[x1p1,x2,x3m1]))
        psi[52] = 0.25*((data[x1,x2p1,x3p2]-data[x1,x2m1,x3p2])
                            -(data[x1,x2p1,x3]-data[x1,x2m1,x3]))
        psi[53] = 0.25*((data[x1p1,x2p1,x3p2]-data[x1p1,x2m1,x3p2])
                            -(data[x1p1,x2p1,x3]-data[x1p1,x2m1,x3]))
        psi[54] = 0.25*((data[x1,x2p2,x3p2]-data[x1,x2,x3p2])
                            -(data[x1,x2p2,x3]-data[x1,x2,x3]))
        psi[55] = 0.25*((data[x1p2,x2p2,x3p2]-data[x1p2,x2,x3p2])
                            -(data[x1p2,x2p2,x3]-data[x1p2,x2,x3]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _set_third_drvtv_(self,int x1,int x1m1,int x1p1,int x1p2,
                                int x2,int x2m1,int x2p1,int x2p2,
                                int x3,int x3m1,int x3p1,int x3p2):
        cdef:
            double *psi = &self.psi[0]
            double[:,:,::1] data = self.data
        # Values of d3f/dxdydz at the corners of the voxel
        psi[56] = 0.125*(((data[x1p1,x2p1,x3p1]-data[x1m1,x2p1,x3p1])
                                -(data[x1p1,x2m1,x3p1]-data[x1m1,x2m1,x3p1]))
                            -((data[x1p1,x2p1,x3m1]-data[x1m1,x2p1,x3m1])
                                -(data[x1p1,x2m1,x3m1]-data[x1m1,x2m1,x3m1])))
        psi[57] = 0.125*(((data[x1p2,x2p1,x3p1]-data[x1,x2p1,x3p1])
                                -(data[x1p2,x2m1,x3p1]-data[x1,x2m1,x3p1]))
                            -((data[x1p2,x2p1,x3m1]-data[x1,x2p1,x3m1])
                                -(data[x1p2,x2m1,x3m1]-data[x1,x2m1,x3m1])))
        psi[58] = 0.125*(((data[x1p1,x2p2,x3p1]-data[x1m1,x2p2,x3p1])
                                -(data[x1p1,x2,x3p1]-data[x1m1,x2,x3p1]))
                            -((data[x1p1,x2p2,x3m1]-data[x1m1,x2p2,x3m1])
                                -(data[x1p1,x2,x3m1]-data[x1m1,x2,x3m1])))
        psi[59] = 0.125*(((data[x1p2,x2p2,x3p1]-data[x1,x2p2,x3p1])
                                -(data[x1p2,x2,x3p1]-data[x1,x2,x3p1]))
                            -((data[x1p2,x2p2,x3m1]-data[x1,x2p2,x3m1])
                                -(data[x1p2,x2,x3m1]-data[x1,x2,x3m1])))
        psi[60] = 0.125*(((data[x1p1,x2p1,x3p2]-data[x1m1,x2p1,x3p2])
                                -(data[x1p1,x2m1,x3p2]-data[x1m1,x2m1,x3p2]))
                            -((data[x1p1,x2p1,x3]-data[x1m1,x2p1,x3])
                                -(data[x1p1,x2m1,x3]-data[x1m1,x2m1,x3])))
        psi[61] = 0.125*(((data[x1p2,x2p1,x3p2]-data[x1,x2p1,x3p2])
                                -(data[x1p2,x2m1,x3p2]-data[x1,x2m1,x3p2]))
                            -((data[x1p2,x2p1,x3]-data[x1,x2p1,x3])
                                -(data[x1p2,x2m1,x3]-data[x1,x2m1,x3])))
        psi[62] = 0.125*(((data[x1p1,x2p2,x3p2]-data[x1m1,x2p2,x3p2])
                                -(data[x1p1,x2,x3p2]-data[x1m1,x2,x3p2]))
                            -((data[x1p1,x2p2,x3]-data[x1m1,x2,x3])
                                -(data[x1p1,x2,x3]-data[x1m1,x2,x3])))
        psi[63] = 0.125*(((data[x1p2,x2p2,x3p2]-data[x1,x2p2,x3p2])
                                -(data[x1p2,x2,x3p2]-data[x1,x2,x3p2]))
                            -((data[x1p2,x2p2,x3]-data[x1,x2p2,x3])
                                -(data[x1p2,x2,x3]-data[x1,x2,x3])))
