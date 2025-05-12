#include <math.h>

#include "misfit_xcorr.h"


void xcorr_cpu(Float64 *result, Int64 L, Int64 Ix, Float64* x, Int64 Iy, Float64* y){
    Int64 i;
    *result = 0.0;
    for(i=0; i<L; i++){
        *result += x[Ix + i] * y[Iy + i];
    }
}

void normalized_xcorr_cpu(Float64 *result, Int64 L, Int64 Ix, Float64* x, Int64 Iy, Float64* y){
    Float64 norm_x, norm_y;
    xcorr_cpu(&norm_x, L, Ix, x, Ix, x);
    xcorr_cpu(&norm_y, L, Iy, y, Iy, y);
    xcorr_cpu(result, L, Ix, x, Iy, y);
    *result /= sqrt(norm_x * norm_y);
}

__device__
void xcorr_gpu(Float64* result, Int64 L, Int64 Ix, Float64* x, Int64 Iy, Float64* y){
    Int64 i;
    *result = 0.0;
    for(i=0; i<L; i++){
        *result += x[Ix + i] * y[Iy + i];
    }
}

__device__
void normalized_xcorr_gpu(Float64 *result, Int64 L, Int64 Ix, Float64* x, Int64 Iy, Float64* y){
    Float64 norm_x, norm_y;
    xcorr_gpu(&norm_x, L, Ix, x, Ix, x);
    xcorr_gpu(&norm_y, L, Iy, y, Iy, y);
    xcorr_gpu(result, L, Ix, x, Iy, y);
    *result /= sqrt(norm_x * norm_y);
}
