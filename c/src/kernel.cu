#define _USE_MATH_DEFINES
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

#define sind(x) sin((x) * M_PI / 180.0)
#define cosd(x) cos((x) * M_PI / 180.0)

__device__ void dc2ts(Float64 *mt11, Float64 *mt22, Float64 *mt33, Float64 *mt12, Float64 *mt13, Float64 *mt23,
                      Float64 s, Float64 d, Float64 r)
{
    Float64 ss = sind(s), cs = cosd(s), s2s = sind(2.0 * s), c2s = cosd(2.0 * s);
    Float64 sd = sind(d), cd = cosd(d), s2d = sind(2.0 * d), c2d = cosd(2.0 * d);
    Float64 sr = sind(r), cr = cosd(r), s2r = sind(2.0 * r), c2r = cosd(2.0 * r);

    *mt11 = -(s2s * sd * cr + (ss * ss) * s2d * sr);
    *mt22 = (s2s * sd * cr - (cs * cs) * s2d * sr);
    *mt33 = s2d * sr;
    *mt12 = (c2s * sd * cr + 0.5 * s2s * s2d * sr);
    *mt13 = -(cs * cd * cr + ss * c2d * sr);
    *mt23 = -(ss * cd * cr - cs * c2d * sr);
    return;
}

void dc2ts_omp(Float64 *mt11, Float64 *mt22, Float64 *mt33, Float64 *mt12, Float64 *mt13, Float64 *mt23,
               Float64 s, Float64 d, Float64 r)
{
    Float64 ss = sind(s), cs = cosd(s), s2s = sind(2.0 * s), c2s = cosd(2.0 * s);
    Float64 sd = sind(d), cd = cosd(d), s2d = sind(2.0 * d), c2d = cosd(2.0 * d);
    Float64 sr = sind(r), cr = cosd(r), s2r = sind(2.0 * r), c2r = cosd(2.0 * r);

    *mt11 = -(s2s * sd * cr + (ss * ss) * s2d * sr);
    *mt22 = (s2s * sd * cr - (cs * cs) * s2d * sr);
    *mt33 = s2d * sr;
    *mt12 = (c2s * sd * cr + 0.5 * s2s * s2d * sr);
    *mt13 = -(cs * cd * cr + ss * c2d * sr);
    *mt23 = -(ss * cd * cr - cs * c2d * sr);
    return;
}

// #define calsyn(v, mt)

#define lin2cart(v, n) \
    do                 \
    {                  \
        v = res % (n); \
        res /= (n);    \
    } while (0)

__global__ void kernel_gpu(GlobalSetting *gs, Record *rs, GreenFunction *gfs, Phase *ps,
                           Float64 *mis_waveform, Int64 *mis_shift, Float64 *mis_pol, Float64 *mis_psr)
{
    Int64 idx, ifreq, iphase, istrike, idip, irake, res;

    idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= (gs->n_frequency_pair * gs->n_phase * gs->nstrike * gs->ndip * gs->nrake)){
        // printf("(kernel_gpu) out of bounds %lld, nfreq: %lld, nphase: %lld\n", idx, gs->n_frequency_pair, gs->n_phase);
        return;
    }


    res = idx;
    lin2cart(ifreq, gs->n_frequency_pair);
    lin2cart(iphase, gs->n_phase);
    lin2cart(istrike, gs->nstrike);
    lin2cart(idip, gs->ndip);
    lin2cart(irake, gs->nrake);

    // if (res)
    // {
    //     mis_waveform[idx] = 12345.0;
    //     mis_shift[idx] = 12345;
    //     mis_pol[idx] = 12345.0;
    //     mis_psr[idx] = 12345.0;
    //     return;
    // }
    mis_waveform[idx] = 20.0;
    mis_shift[idx] = idx;
    mis_pol[idx] = 20.0;
    mis_psr[idx] = 20.0;
    return;

    Float64 strike, dip, rake, mt11, mt22, mt33, mt12, mt13, mt23;

    strike = istrike * gs->dstrike;
    dip = idip * gs->ddip;
    rake = irake * gs->drake - 90.0;

    dc2ts(&mt11, &mt22, &mt33, &mt12, &mt13, &mt23, strike, dip, rake);

    Int64 igf, irec;
    for (irec = 0; irec < gs->n_record; irec++)
        if (rs[irec].id == ps[iphase].rid)
            break;
    for (igf = 0; igf < gs->n_record * gs->n_event_location; igf++)
        if ((gfs[igf].rid == ps[iphase].rid) && (gfs[igf].eid == ps[iphase].eid))
            break;
    Float64 nxc;
    Int64 datshift;
    datshift = ifreq * rs[irec].npts;
    maximum_xcorr_mt_gpu(&nxc, &mis_shift[idx], ps[iphase].length, 10,
                         rs[irec].npts, ps[iphase].Rstart, &(rs[irec].data[datshift]),
                         rs[irec].npts, ps[iphase].Estart,
                         &(gfs[igf].g11[datshift]), &(gfs[igf].g22[datshift]), &(gfs[igf].g33[datshift]),
                         &(gfs[igf].g12[datshift]), &(gfs[igf].g13[datshift]), &(gfs[igf].g23[datshift]),
                         mt11, mt22, mt33, mt12, mt13, mt23);
    mis_waveform[idx] = sqrt((1.0 - nxc) * 0.5);
    mis_pol[idx] = 0.0;
    mis_psr[idx] = 0.0;
    return;
}

void kernel_omp(Int64 idx, GlobalSetting *gs, Record *rs, GreenFunction *gfs, Phase *ps,
                Float64 *mis_waveform, Int64 *mis_shift, Float64 *mis_pol, Float64 *mis_psr)
{
    // Int64 idx;
    Int64 ifreq, iphase, istrike, idip, irake, res;

    // idx = threadIdx.x + (blockIdx.x * blockDim.x);

    res = idx;
    lin2cart(ifreq, gs->n_frequency_pair);
    lin2cart(iphase, gs->n_phase);
    lin2cart(istrike, gs->nstrike);
    lin2cart(idip, gs->ndip);
    lin2cart(irake, gs->nrake);

    // if (res)
    // {
    //     mis_waveform[idx] = 12345.0;
    //     mis_shift[idx] = 12345;
    //     mis_pol[idx] = 12345.0;
    //     mis_psr[idx] = 12345.0;
    //     return;
    // }
    mis_waveform[idx] = 20.0;
    mis_shift[idx] = idx;
    mis_pol[idx] = 20.0;
    mis_psr[idx] = 20.0;
    return;

    Float64 strike, dip, rake, mt11, mt22, mt33, mt12, mt13, mt23;

    strike = istrike * gs->dstrike;
    dip = idip * gs->ddip;
    rake = irake * gs->drake - 90.0;

    dc2ts_omp(&mt11, &mt22, &mt33, &mt12, &mt13, &mt23, strike, dip, rake);

    Int64 igf, irec;
    for (irec = 0; irec < gs->n_record; irec++)
        if (rs[irec].id == ps[iphase].rid)
            break;
    for (igf = 0; igf < gs->n_record * gs->n_event_location; igf++)
        if ((gfs[igf].rid == ps[iphase].rid) && (gfs[igf].eid == ps[iphase].eid))
            break;
    Float64 nxc;
    Int64 datshift;
    datshift = ifreq * rs[irec].npts;
    maximum_xcorr_mt_cpu(&nxc, &mis_shift[idx], ps[iphase].length, 10,
                         rs[irec].npts, ps[iphase].Rstart, &(rs[irec].data[datshift]),
                         rs[irec].npts, ps[iphase].Estart,
                         &(gfs[igf].g11[datshift]), &(gfs[igf].g22[datshift]), &(gfs[igf].g33[datshift]),
                         &(gfs[igf].g12[datshift]), &(gfs[igf].g13[datshift]), &(gfs[igf].g23[datshift]),
                         mt11, mt22, mt33, mt12, mt13, mt23);
    mis_waveform[idx] = sqrt((1.0 - nxc) * 0.5);
    mis_pol[idx] = 0.0;
    mis_psr[idx] = 0.0;
    return;
}

static inline Int64 _max_(Int64 a, Int64 b) { return (a > b) ? a : b; }

void call_kernel_gpu(GlobalSetting_xPU *gs, Record_xPU *rlist, GreenFunction_xPU *gflist, Phase_xPU *plist, Result_xPU *result)
{
    Int64 n_threads;
    n_threads = gs->cpu->n_frequency_pair * gs->cpu->n_phase * gs->cpu->nstrike * gs->cpu->ndip * gs->cpu->nrake;
    int block = 32;
    int grid = (n_threads - 1) / 32 + 1;
    Result_xPU_sync(result);
    kernel_gpu<<<grid, block>>>(gs->gpu, rlist->gpu, gflist->gpu, plist->gpu,
                                result->waveform_gpu, result->shift_gpu, result->polarity_gpu, result->ps_ratio_gpu);
    cudaDeviceSynchronize();
    result->mgpu = _max_(result->mcpu, result->mgpu) + 1;
    Result_xPU_sync(result);
}

void call_kernel_omp(GlobalSetting_xPU *gs, Record_xPU *rlist, GreenFunction_xPU *gflist, Phase_xPU *plist, Result_xPU *result)
{
    Int64 n_threads, idx;
    n_threads = gs->cpu->n_frequency_pair * gs->cpu->n_phase * gs->cpu->nstrike * gs->cpu->ndip * gs->cpu->nrake;
    Result_xPU_sync(result);

#pragma omp parallel for num_threads(12) default(none) shared(gs, rlist, gflist, plist, result) private(idx)
    for (idx = 0; idx < n_threads; idx++)
        kernel_omp(idx, gs->cpu, rlist->cpu, gflist->cpu, plist->cpu,
                   result->waveform, result->shift, result->polarity, result->ps_ratio);

    result->mcpu = _max_(result->mcpu, result->mgpu) + 1;
    Result_xPU_sync(result);
}
