ctypedef fused number:
    int
    double
    long


cdef number c_min(number[:] arr)

cdef number c_sum(number[:,:] arr)

cdef number[:] c_sum_axis_0(number[:,:] arr)

cdef number[:] c_sum_axis_1(number[:,:] arr)
