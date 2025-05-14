#ifndef __GREENFUNCTION_H__
#define __GREENFUNCTION_H__

#include "types.h"
#include "GlobalSetting.h"
#include "Record.h"

typedef struct _GreenFunction_t
{
    Int64 rid, eid;
    Float64 *g11, *g22, *g33, *g12, *g13, *g23;
} GreenFunction;

void GreenFunction_init(GreenFunction *gf);
void GreenFunction_free(GreenFunction *gf);
void GreenFunction_read(GreenFunction *gf, Record *rs, GlobalSetting *gs, FILE *fp);
void GreenFunction_write(GreenFunction *gf, Record *rs, GlobalSetting *gs, FILE *fp);

void GreenFunction_copyto_gpu(GreenFunction *gf_cpu, GreenFunction *gf_gpu, Int64 npts, Int64 nfreq);
void GreenFunction_copy_gpu_to_cpu(GreenFunction *gf_cpu, GreenFunction *gf_gpu, Int64 npts, Int64 nfreq);
void GreenFunction_free_gpu(GreenFunction *gf);

typedef struct _GreenFunction_xPU_t
{
    Int64 n, mcpu, mgpu;
    GreenFunction *cpu, *gpu;
} GreenFunction_xPU;

void GreenFunction_xPU_alloc(GreenFunction_xPU *gflist, Int64 n);
void GreenFunction_xPU_free(GreenFunction_xPU *gflist);
void GreenFunction_xPU_read(GreenFunction_xPU *gflist, Record_xPU *rlist, GlobalSetting_xPU *gs, FILE *fp);
void GreenFunction_xPU_write(GreenFunction_xPU *gflist, Record_xPU *rlist, GlobalSetting_xPU *gs, FILE *fp);
void GreenFunction_xPU_sync(GreenFunction_xPU *gflist, Record_xPU *rlist, GlobalSetting_xPU *gs);

#endif // __GREENFUNCTION_H__
