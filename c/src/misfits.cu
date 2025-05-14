#include <math.h>
#include "types.h"
#include "misfits.h"

void dot32_cpu(Float32 *result, Int32 L, Int32 Ix, Float32 *x, Int32 Iy, Float32 *y)
{
    Int32 i;
    *result = 0.0;
    for (i = 0; i < L; i++)
    {
        *result += x[Ix + i] * y[Iy + i];
    }
}

void dot64_cpu(Float64 *result, Int64 L, Int64 Ix, Float64 *x, Int64 Iy, Float64 *y)
{
    Int64 i;
    *result = 0.0;
    for (i = 0; i < L; i++)
    {
        *result += x[Ix + i] * y[Iy + i];
    }
}

void normalized_xcorr32_mt_cpu(Float32 *result, Int32 L, Int32 Ix, Float32 *obs, Int32 Iy,
                               Float32 *g11, Float32 *g22, Float32 *g33, Float32 *g12, Float32 *g13, Float32 *g23,
                               Float32 m11, Float32 m22, Float32 m33, Float32 m12, Float32 m13, Float32 m23)
{
    Int32 i;
    Float32 w, a, b;
    *result = 0.0;
    a = 0.0;
    b = 0.0;
    for (i = 0; i < L; i++)
    {
        w = g11[Iy + i] * m11 + g22[Iy + i] * m22 + g33[Iy + i] * m33 + g12[Iy + i] * m12 + g13[Iy + i] * m13 + g23[Iy + i] * m23;
        a += w * w;
        b += obs[Ix + i] * obs[Ix + i];
        *result += w * obs[Ix + i];
    }
    *result /= sqrt(a * b);
}

void normalized_xcorr64_mt_cpu(Float64 *result, Int64 L, Int64 Ix, Float64 *obs, Int64 Iy,
                               Float64 *g11, Float64 *g22, Float64 *g33, Float64 *g12, Float64 *g13, Float64 *g23,
                               Float64 m11, Float64 m22, Float64 m33, Float64 m12, Float64 m13, Float64 m23)
{
    Int64 i;
    Float64 w, a, b;
    *result = 0.0;
    a = 0.0;
    b = 0.0;
    for (i = 0; i < L; i++)
    {
        w = g11[Iy + i] * m11 + g22[Iy + i] * m22 + g33[Iy + i] * m33 + g12[Iy + i] * m12 + g13[Iy + i] * m13 + g23[Iy + i] * m23;
        a += w * w;
        b += obs[Ix + i] * obs[Ix + i];
        *result += w * obs[Ix + i];
    }
    *result /= sqrt(a * b);
}

void normalized_xcorr32_cpu(Float32 *result, Int32 L, Int32 Ix, Float32 *x, Int32 Iy, Float32 *y)
{
    Float32 norm_x, norm_y;
    dot32_cpu(&norm_x, L, Ix, x, Ix, x);
    dot32_cpu(&norm_y, L, Iy, y, Iy, y);
    dot32_cpu(result, L, Ix, x, Iy, y);
    *result /= sqrtf(norm_x * norm_y);
}

void normalized_xcorr64_cpu(Float64 *result, Int64 L, Int64 Ix, Float64 *x, Int64 Iy, Float64 *y)
{
    Float64 norm_x, norm_y;
    dot64_cpu(&norm_x, L, Ix, x, Ix, x);
    dot64_cpu(&norm_y, L, Iy, y, Iy, y);
    dot64_cpu(result, L, Ix, x, Iy, y);
    *result /= sqrt(norm_x * norm_y);
}

void maximum_xcorr_cpu(Float64 *maxcorr, Int64 *shift, Int64 L, Int64 S,
                       Int64 Lx, Int64 Ix, Float64 *x, Int64 Ly, Int64 Iy, Float64 *y)
{
    Int64 s;
    Float64 v;
    *maxcorr = -2.0;
    for (s = -S; s <= S; s++)
    {
        normalized_xcorr64_cpu(&v, L, Ix, x, Iy - s, y);
        if (v > *maxcorr)
        {
            *maxcorr = v;
            *shift = s;
        }
    }
}

void maximum_xcorr_mt_cpu(Float64 *maxcorr, Int64 *shift, Int64 L, Int64 S,
                          Int64 Lx, Int64 Ix, Float64 *obs, Int64 Ly, Int64 Iy,
                          Float64 *g11, Float64 *g22, Float64 *g33, Float64 *g12, Float64 *g13, Float64 *g23,
                          Float64 m11, Float64 m22, Float64 m33, Float64 m12, Float64 m13, Float64 m23)
{
    Int64 s;
    Float64 v;
    *maxcorr = -2.0;
    for (s = -S; s <= S; s++)
    {
        normalized_xcorr64_mt_cpu(&v, L, Ix, obs, Iy - s, g11, g22, g33, g12, g13, g23, m11, m22, m33, m12, m13, m23);
        if (v > *maxcorr)
        {
            *maxcorr = v;
            *shift = s;
        }
    }
}

__device__ void dot32_gpu(Float32 *result, Int32 L, Int32 Ix, Float32 *x, Int32 Iy, Float32 *y)
{
    Int32 i;
    *result = 0.0;
    for (i = 0; i < L; i++)
    {
        *result += x[Ix + i] * y[Iy + i];
    }
}

__device__ void dot64_gpu(Float64 *result, Int64 L, Int64 Ix, Float64 *x, Int64 Iy, Float64 *y)
{
    Int64 i;
    *result = 0.0;
    for (i = 0; i < L; i++)
    {
        *result += x[Ix + i] * y[Iy + i];
    }
}

__device__ void normalized_xcorr32_gpu(Float32 *result, Int32 L, Int32 Ix, Float32 *x, Int32 Iy, Float32 *y)
{
    Float32 norm_x, norm_y;
    dot32_gpu(&norm_x, L, Ix, x, Ix, x);
    dot32_gpu(&norm_y, L, Iy, y, Iy, y);
    dot32_gpu(result, L, Ix, x, Iy, y);
    *result /= sqrtf(norm_x * norm_y);
}

__device__ void normalized_xcorr64_gpu(Float64 *result, Int64 L, Int64 Ix, Float64 *x, Int64 Iy, Float64 *y)
{
    Float64 norm_x, norm_y;
    dot64_gpu(&norm_x, L, Ix, x, Ix, x);
    dot64_gpu(&norm_y, L, Iy, y, Iy, y);
    dot64_gpu(result, L, Ix, x, Iy, y);
    *result /= sqrt(norm_x * norm_y);
}

__device__ void maximum_xcorr_gpu(Float64 *maxcorr, Int64 *shift, Int64 L, Int64 S,
                                  Int64 Lx, Int64 Ix, Float64 *x, Int64 Ly, Int64 Iy, Float64 *y)
{
    Int64 s;
    Float64 v;
    *maxcorr = -2.0;
    for (s = -S; s <= S; s++)
    {
        normalized_xcorr64_gpu(&v, L, Ix, x, Iy - s, y);
        if (v > *maxcorr)
        {
            *maxcorr = v;
            *shift = s;
        }
    }
}

__device__ void normalized_xcorr32_mt_gpu(Float32 *result, Int32 L, Int32 Ix, Float32 *obs, Int32 Iy,
                                          Float32 *g11, Float32 *g22, Float32 *g33, Float32 *g12, Float32 *g13, Float32 *g23,
                                          Float32 m11, Float32 m22, Float32 m33, Float32 m12, Float32 m13, Float32 m23)
{
    Int32 i;
    Float32 w, a, b;
    *result = 0.0;
    a = 0.0;
    b = 0.0;
    for (i = 0; i < L; i++)
    {
        w = g11[Iy + i] * m11 + g22[Iy + i] * m22 + g33[Iy + i] * m33 + g12[Iy + i] * m12 + g13[Iy + i] * m13 + g23[Iy + i] * m23;
        a += w * w;
        b += obs[Ix + i] * obs[Ix + i];
        *result += w * obs[Ix + i];
    }
    *result /= sqrt(a * b);
}

__device__ void normalized_xcorr64_mt_gpu(Float64 *result, Int64 L, Int64 Ix, Float64 *obs, Int64 Iy,
                                          Float64 *g11, Float64 *g22, Float64 *g33, Float64 *g12, Float64 *g13, Float64 *g23,
                                          Float64 m11, Float64 m22, Float64 m33, Float64 m12, Float64 m13, Float64 m23)
{
    Int64 i;
    Float64 w, a, b;
    *result = 0.0;
    a = 0.0;
    b = 0.0;
    for (i = 0; i < L; i++)
    {
        w = g11[Iy + i] * m11 + g22[Iy + i] * m22 + g33[Iy + i] * m33 + g12[Iy + i] * m12 + g13[Iy + i] * m13 + g23[Iy + i] * m23;
        a += w * w;
        b += obs[Ix + i] * obs[Ix + i];
        *result += w * obs[Ix + i];
    }
    *result /= sqrt(a * b);
}

__device__ void maximum_xcorr_mt_gpu(Float64 *maxcorr, Int64 *shift, Int64 L, Int64 S,
                                     Int64 Lx, Int64 Ix, Float64 *obs, Int64 Ly, Int64 Iy,
                                     Float64 *g11, Float64 *g22, Float64 *g33, Float64 *g12, Float64 *g13, Float64 *g23,
                                     Float64 m11, Float64 m22, Float64 m33, Float64 m12, Float64 m13, Float64 m23)
{
    Int64 s;
    Float64 v;
    *maxcorr = -2.0;
    for (s = -S; s <= S; s++)
    {
        normalized_xcorr64_mt_gpu(&v, L, Ix, obs, Iy - s, g11, g22, g33, g12, g13, g23, m11, m22, m33, m12, m13, m23);
        if (v > *maxcorr)
        {
            *maxcorr = v;
            *shift = s;
        }
    }
}
