#include <math.h>
#include <cuda.h>

#include "types.h"
#include "GlobalSetting.h"
#include "Record.h"
#include "GreenFunction.h"
#include "Phase.h"
#include "Result.h"
#include "misfits.h"
#include "kernel.h"

__device__ void dc2ts(Float64 *mt11, Float64 *mt22, Float64 *mt33, Float64 *mt12, Float64 *mt13, Float64 *mt23,
                      Float64 s, Float64 d, Float64 r) {}

// #define calsyn(v, mt)

__global__ void kernel_gpu(GlobalSetting *gs, Record *rs, GreenFunction *gfs, Phase *ps,
                           Float64 *mis_waveform, Int64 *mis_shift, Float64 *mis_pol, Float64 *mis_psr)
{
    Int64 idx, ip, istrike, idip, irake, res;
    idx = threadIdx.x + (blockIdx.x * blockDim.x);
    res = idx;
    ip = res % gs->n_phase;
    res /= gs->n_phase;
    irake = res % gs->nrake;
    res /= gs->nrake;
    idip = res % gs->ndip;
    res /= gs->ndip;
    istrike = res % gs->nstrike;
    res /= gs->nstrike;

    // res == 0 ?

    Float64 strike = istrike * gs->dstrike, dip = idip * gs->ddip, rake = irake * gs->drake - 180.0;
    Float64 mt11, mt22, mt33, mt12, mt13, mt23;
    dc2ts(&mt11, &mt22, &mt33, &mt12, &mt13, &mt23, strike, dip, rake);

    Int64 igf, irec;
    for (irec = 0; irec < gs->n_record; irec++)
        if (rs[irec].id == ps[ip].rid)
            break;
    for (igf = 0; igf < gs->n_record * gs->n_event_location; igf++)
        if ((gfs[igf].rid == ps[ip].rid) && (gfs[igf].eid == ps[ip].eid))
            break;
    Float64 nxc;
    maximum_xcorr_mt_gpu(&nxc, &mis_shift[idx], 10, 5, rs[irec].npts, ps[ip].Rstart, rs[irec].data,
                         rs[irec].npts, ps[ip].Estart, gfs[igf].g11, gfs[igf].g22, gfs[igf].g33, gfs[igf].g12, gfs[igf].g13, gfs[igf].g23,
                         mt11, mt22, mt33, mt12, mt13, mt23);
    mis_waveform[idx] = sqrt((1.0 - nxc) * 0.5);
    mis_pol[idx] = 0.0;
    mis_psr[idx] = 0.0;
    return;
}

static inline Int64 _max_(Int64 a, Int64 b) { return (a > b) ? a : b; }

void kernel(GlobalSetting_xPU *gs, Record_xPU *rlist, GreenFunction_xPU *gflist, Phase_xPU *plist, Result_xPU *result)
{
    Int64 n_threads;
    n_threads = gs->cpu->n_phase * gs->cpu->nstrike * gs->cpu->ndip * gs->cpu->nrake;
    dim3 block(16);
    dim3 grid((n_threads - 1) / 16 + 1);
    kernel_gpu<<<grid, block>>>(gs->gpu, rlist->gpu, gflist->gpu, plist->gpu,
                                result->waveform_gpu, result->shift_gpu, result->polarity_gpu, result->ps_ratio_gpu);
    result->mgpu = _max_(result->mcpu, result->mgpu) + 1;
    cudaDeviceSynchronize();
}
