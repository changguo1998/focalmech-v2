
#include <stdio.h>
#include <cuda.h>
#include "types.h"
#include "Result.h"

void Result_xPU_alloc(Result_xPU *res, Int64 n_phase, Int64 n_fm)
{
    Int64 n = n_phase * n_fm;
    res->n_phase = n_phase;
    res->n_fm = n_fm;
    res->mcpu = 0;
    res->mgpu = 0;
    res->waveform = (Float64 *)malloc(n * sizeof(Float64));
    res->shift = (Int64 *)malloc(n * sizeof(Int64));
    res->polarity = (Float64 *)malloc(n * sizeof(Float64));
    res->ps_ratio = (Float64 *)malloc(n * sizeof(Float64));
    cudaMalloc(&(res->waveform_gpu), n * sizeof(Float64));
    cudaMalloc(&(res->shift_gpu), n * sizeof(Int64));
    cudaMalloc(&(res->polarity_gpu), n * sizeof(Float64));
    cudaMalloc(&(res->ps_ratio_gpu), n * sizeof(Float64));
}

#define _R_FREE_MAT(var)                \
    if (res->##var)                     \
        do                              \
        {                               \
            free(res->##var);           \
            res->##var = NULL;          \
        } while (0);                    \
    if (res->##var##_gpu)               \
        do                              \
        {                               \
            cudaFree(res->##var##_gpu); \
            res->##var##_gpu = NULL;    \
    } while (0)

void Result_xPU_free(Result_xPU *res)
{
    _R_FREE_MAT(waveform);
    _R_FREE_MAT(shift);
    _R_FREE_MAT(polarity);
    _R_FREE_MAT(ps_ratio);
}

#undef _R_FREE_MAT

Int64 _max_(Int64 a, Int64 b) { return (a > b) ? a : b; }

void Result_xPU_sync(Result_xPU *res)
{
    Int64 n;
    n = res->n_phase * res->n_fm;
    if (res->mcpu > res->mgpu)
    {
        cudaMemcpy(res->waveform_gpu, res->waveform, n * sizeof(Float64), cudaMemcpyHostToDevice);
        cudaMemcpy(res->shift_gpu, res->shift, n * sizeof(Int64), cudaMemcpyHostToDevice);
        cudaMemcpy(res->polarity_gpu, res->polarity, n * sizeof(Float64), cudaMemcpyHostToDevice);
        cudaMemcpy(res->ps_ratio_gpu, res->ps_ratio, n * sizeof(Float64), cudaMemcpyHostToDevice);
    }
    if (res->mcpu < res->mgpu)
    {
        cudaMemcpy(res->waveform, res->waveform_gpu, n * sizeof(Float64), cudaMemcpyDeviceToHost);
        cudaMemcpy(res->shift, res->shift_gpu, n * sizeof(Int64), cudaMemcpyDeviceToHost);
        cudaMemcpy(res->polarity, res->polarity_gpu, n * sizeof(Float64), cudaMemcpyDeviceToHost);
        cudaMemcpy(res->ps_ratio, res->ps_ratio_gpu, n * sizeof(Float64), cudaMemcpyDeviceToHost);
    }
    res->mcpu = 0;
    res->mgpu = 0;
    cudaDeviceSynchronize();
}

void Result_xPU_read(Result_xPU *res, FILE *fp)
{
    fread(&res->n_phase, sizeof(Int64), 1, fp);
    fread(&res->n_fm, sizeof(Int64), 1, fp);
    Int64 n = res->n_phase * res->n_fm;
    Result_xPU_alloc(res, res->n_phase, res->n_fm);
    fread(res->waveform, sizeof(Float64), n, fp);
    fread(res->shift, sizeof(Int64), n, fp);
    fread(res->polarity, sizeof(Float64), n, fp);
    fread(res->ps_ratio, sizeof(Float64), n, fp);
    res->mcpu = _max_(res->mcpu, res->mgpu) + 1;
    Result_xPU_sync(res);
}

void Result_xPU_write(Result_xPU *res, FILE *fp)
{
    Result_xPU_sync(res);
    Int64 n = res->n_phase * res->n_fm;
    fwrite(&res->n_phase, sizeof(Int64), 1, fp);
    fwrite(&res->n_fm, sizeof(Int64), 1, fp);
    fwrite(res->waveform, sizeof(Float64), n, fp);
    fwrite(res->shift, sizeof(Int64), n, fp);
    fwrite(res->polarity, sizeof(Float64), n, fp);
    fwrite(res->ps_ratio, sizeof(Float64), n, fp);
}
