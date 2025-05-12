#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "GreenFunction.h"

void GreenFunction_init(GreenFunction *gf)
{
    gf->rid = 0;
    gf->eid = 0;
    gf->g11 = NULL;
    gf->g22 = NULL;
    gf->g33 = NULL;
    gf->g12 = NULL;
    gf->g13 = NULL;
    gf->g23 = NULL;
}

#define FREE_MEMORY(var) \
    if (var)             \
        do               \
        {                \
            free(var);   \
            var = NULL;  \
    } while (0)

void GreenFunction_free(GreenFunction *gf)
{
    FREE_MEMORY(gf->g11);
    FREE_MEMORY(gf->g22);
    FREE_MEMORY(gf->g33);
    FREE_MEMORY(gf->g12);
    FREE_MEMORY(gf->g13);
    FREE_MEMORY(gf->g23);
}

#undef FREE_MEMORY

#define GF_READ_MAT(var)                             \
    var = (Float64 *)malloc(ndat * sizeof(Float64)); \
    fread(var, sizeof(Float64), ndat, fp)

void GreenFunction_read(GreenFunction *gf, Record *rs, GlobalSetting *gs, FILE *fp)
{
    fread(&gf->rid, sizeof(Int64), 1, fp);
    fread(&gf->eid, sizeof(Int64), 1, fp);
    Int64 i, npts;
    for (i = 0; i < gs->n_record; i++)
    {
        if (rs[i].id == gf->rid)
        {
            npts = rs[i].npts;
            break;
        }
    }
    Int64 ndat = npts * gs->n_frequency_pair;
    GF_READ_MAT(gf->g11);
    GF_READ_MAT(gf->g22);
    GF_READ_MAT(gf->g33);
    GF_READ_MAT(gf->g12);
    GF_READ_MAT(gf->g13);
    GF_READ_MAT(gf->g23);
}

#undef GF_READ_MAT

void GreenFunction_write(GreenFunction *gf, Record *rs, GlobalSetting *gs, FILE *fp)
{
    Int64 i, npts;
    for (i = 0; i < gs->n_record; i++)
    {
        if (rs[i].id == gf->rid)
        {
            npts = rs[i].npts;
            break;
        }
    }
    Int64 ndat = npts * gs->n_frequency_pair;
    fwrite(&gf->rid, sizeof(Int64), 1, fp);
    fwrite(&gf->eid, sizeof(Int64), 1, fp);
    fwrite(gf->g11, sizeof(Float64), ndat, fp);
    fwrite(gf->g22, sizeof(Float64), ndat, fp);
    fwrite(gf->g33, sizeof(Float64), ndat, fp);
    fwrite(gf->g12, sizeof(Float64), ndat, fp);
    fwrite(gf->g13, sizeof(Float64), ndat, fp);
    fwrite(gf->g23, sizeof(Float64), ndat, fp);
}

#define _GF_COPY_MAT(var)                                   \
    cudaMalloc(&gbuf.##var, npts *nfreq * sizeof(Float64)); \
    cudaMemcpy(gbuf.##var, gf_cpu->##var, npts *nfreq * sizeof(Float64), cudaMemcpyHostToDevice)

void GreenFunction_copyto_gpu(GreenFunction *gf_cpu, GreenFunction *gf_gpu, Int64 npts, Int64 nfreq)
{
    GreenFunction gbuf;
    gbuf.eid = gf_cpu->eid;

    _GF_COPY_MAT(g11);
    _GF_COPY_MAT(g22);
    _GF_COPY_MAT(g33);
    _GF_COPY_MAT(g12);
    _GF_COPY_MAT(g13);
    _GF_COPY_MAT(g23);
    cudaMemcpy(gf_gpu, &gbuf, sizeof(GreenFunction), cudaMemcpyHostToDevice);
}

#undef _GF_COPY_MAT

#define _GF_FREE_CUDA_MAT(var) cudaFree(gbuf.##var)

void GreenFunction_free_gpu(GreenFunction *gf)
{
    GreenFunction gbuf;
    cudaMemcpy(&gbuf, gf, sizeof(GreenFunction), cudaMemcpyDeviceToHost);
    _GF_FREE_CUDA_MAT(g11);
    _GF_FREE_CUDA_MAT(g22);
    _GF_FREE_CUDA_MAT(g33);
    _GF_FREE_CUDA_MAT(g12);
    _GF_FREE_CUDA_MAT(g13);
    _GF_FREE_CUDA_MAT(g23);
}

#undef _GF_FREE_CUDA_MAT

void GreenFunction_xPU_alloc(GreenFunction_xPU *gflist, Int64 n)
{
    if (n <= 0)
    {
        gflist->cpu = NULL;
        gflist->gpu = NULL;
        gflist->n = 0;
        return;
    }
    gflist->cpu = (GreenFunction *)malloc(n * sizeof(GreenFunction));
    gflist->gpu = NULL;
    gflist->n = n;
}

void GreenFunction_xPU_free(GreenFunction_xPU *gflist)
{
    Int64 i;
    for (i = 0; i < gflist->n; i++)
        GreenFunction_free(&(gflist->cpu[i]));
}

void GreenFunction_xPU_read(GreenFunction_xPU *gflist, Record_xPU *rlist, GlobalSetting_xPU *gs, FILE *fp)
{
    Int64 i;
    for (i = 0; i < gflist->n; i++)
        GreenFunction_read(&(gflist->cpu[i]), rlist->cpu, gs->cpu, fp);
}

void GreenFunction_xPU_write(GreenFunction_xPU *gflist, Record_xPU *rlist, GlobalSetting_xPU *gs, FILE *fp)
{
    Int64 i;
    for (i = 0; i < gflist->n; ++i)
        GreenFunction_write(&gflist->cpu[i], rlist->cpu, gs->cpu, fp);
}
