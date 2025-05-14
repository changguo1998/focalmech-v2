#ifndef __RESULT_H__
#define __RESULT_H__

#include <stdio.h>
#include "types.h"

typedef struct _Result_xPU_t
{
    Int64 n_phase, n_fm, mcpu, mgpu;
    Float64 *waveform, *waveform_gpu;
    Int64 *shift, *shift_gpu;
    Float64 *polarity, *polarity_gpu;
    Float64 *ps_ratio, *ps_ratio_gpu;
} Result_xPU;

void Result_xPU_alloc(Result_xPU *res, Int64 n_phase, Int64 n_fm);
void Result_xPU_free(Result_xPU *res);
void Result_xPU_sync(Result_xPU *res);
void Result_xPU_read(Result_xPU *res, FILE *fp);
void Result_xPU_write(Result_xPU *res, FILE *fp);

#endif // __RESULT_H__
