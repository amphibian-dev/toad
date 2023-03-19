# cython: language_level = 3, infer_types = True, boundscheck = False

import numpy as np
cimport numpy as np
cimport cython



cdef number c_min(number[:] arr):
    cdef number res = np.inf

    for i in range(arr.shape[0]):
        if res > arr[i]:
            res = arr[i]
    return res


cdef number c_sum(number[:,:] arr):
    cdef number res = 0

    cdef Py_ssize_t i,j
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res += arr[i, j]

    return res


cdef number[:] c_sum_axis_0(number[:,:] arr):
    cdef number[:] res = np.zeros(arr.shape[1], dtype=float)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res[j] += arr[i, j]

    return res


cdef number[:] c_sum_axis_1(number[:,:] arr):
    cdef number[:] res = np.zeros(arr.shape[0], dtype=float)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res[i] += arr[i, j]

    return res
