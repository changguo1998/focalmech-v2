#ifndef __MISFIT_NORM2_H__
#define __MISFIT_NORM2_H__

#include "types.h"


#define GPU

void dot_cpu(Float64 *result, Int64 L, Int64 Ix, Float64* x, Int64 Iy, Float64* y);

void normalized_xcorr_cpu(Float64 *result, Int64 L, Int64 Ix, Float64* x, Int64 Iy, Float64* y);

#ifdef GPU
__device__
void xcorr_gpu(Float64* result, Int64 L, Int64 Ix, Float64* x, Int64 Iy, Float64* y);

__device__
void normalized_xcorr_gpu(Float64 *result, Int64 L, Int64 Ix, Float64* x, Int64 Iy, Float64* y);
#endif

#endif // __MISFIT_NORM2_H__
