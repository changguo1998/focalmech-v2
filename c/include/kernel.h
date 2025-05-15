#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "types.h"
#include "GlobalSetting.h"
#include "Record.h"
#include "GreenFunction.h"
#include "Phase.h"
#include "Result.h"
#include "misfits.h"

__global__
void kernel_gpu(GlobalSetting *gs, Record *rs, GreenFunction *gfs, Phase *ps,
    Float64 *mis_waveform, Int64 *mis_shift, Float64 *pol, Float64 *psr);

void kernel_omp(Int64 idx, GlobalSetting *gs, Record *rs, GreenFunction *gfs, Phase *ps,
    Float64 *mis_waveform, Int64 *mis_shift, Float64 *pol, Float64 *psr);

void call_kernel_gpu(GlobalSetting_xPU* gs, Record_xPU *rlist, GreenFunction_xPU *gflist, Phase_xPU *plist, Result_xPU *result);

void call_kernel_omp(GlobalSetting_xPU* gs, Record_xPU *rlist, GreenFunction_xPU *gflist, Phase_xPU *plist, Result_xPU *result);

#endif // __KERNEL_H__
