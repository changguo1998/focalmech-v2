#ifndef __MISFIT_NORM2_H__
#define __MISFIT_NORM2_H__

#include "types.h"

void dot64_cpu(Float64 *result, Int64 L, Int64 Ix, Float64 *x, Int64 Iy, Float64 *y);

void normalized_xcorr64_cpu(Float64 *result, Int64 L, Int64 Ix, Float64 *x, Int64 Iy, Float64 *y);

void normalized_xcorr64_mt_cpu(Float64 *result, Int64 L, Int64 Ix, Float64 *obs, Int64 Iy,
                               Float64 *g11, Float64 *g22, Float64 *g33, Float64 *g12, Float64 *g13, Float64 *g23,
                               Float64 m11, Float64 m22, Float64 m33, Float64 m12, Float64 m13, Float64 m23);

void maximum_xcorr_cpu(Float64 *maxcorr, Int64 *shift, Int64 L, Int64 S,
                       Int64 Lx, Int64 Ix, Float64 *x, Int64 Ly, Int64 Iy, Float64 *y);

void maximum_xcorr_mt_cpu(Float64 *maxcorr, Int64 *shift, Int64 L, Int64 S,
                          Int64 Lx, Int64 Ix, Float64 *obs, Int64 Ly, Int64 Iy,
                          Float64 *g11, Float64 *g22, Float64 *g33, Float64 *g12, Float64 *g13, Float64 *g23,
                          Float64 m11, Float64 m22, Float64 m33, Float64 m12, Float64 m13, Float64 m23);

__device__ void dot64_gpu(Float64 *result, Int64 L, Int64 Ix, Float64 *x, Int64 Iy, Float64 *y);

__device__ void normalized_xcorr64_gpu(Float64 *result, Int64 L, Int64 Ix, Float64 *x, Int64 Iy, Float64 *y);

__device__ void normalized_xcorr64_mt_gpu(Float64 *result, Int64 L, Int64 Ix, Float64 *obs, Int64 Iy,
                                          Float64 *g11, Float64 *g22, Float64 *g33, Float64 *g12, Float64 *g13, Float64 *g23,
                                          Float64 m11, Float64 m22, Float64 m33, Float64 m12, Float64 m13, Float64 m23);

__device__ void maximum_xcorr_mt_gpu(Float64 *maxcorr, Int64 *shift, Int64 L, Int64 S,
                                     Int64 Lx, Int64 Ix, Float64 *obs, Int64 Ly, Int64 Iy,
                                     Float64 *g11, Float64 *g22, Float64 *g33, Float64 *g12, Float64 *g13, Float64 *g23,
                                     Float64 m11, Float64 m22, Float64 m33, Float64 m12, Float64 m13, Float64 m23);
#endif // __MISFIT_NORM2_H__
