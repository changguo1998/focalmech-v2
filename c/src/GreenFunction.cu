#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "GreenFunction.h"

// #define DEBUG

static inline Int64 _max_(Int64 a, Int64 b) { return (a > b) ? a : b; }

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
#ifdef DEBUG
    printf("(GreenFunction_read) start\n");
#endif
    fread(&gf->rid, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(GreenFunction_read) rid: %lld\n", gf->rid);
#endif
    fread(&gf->eid, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(GreenFunction_read) eid: %lld\n", gf->eid);
#endif
    Int64 i, npts;
    Record *rp;
    rp = Record_get_pointer(rs, gs->n_record, gf->rid);
    if (rp == NULL)
    {
        printf("(GreenFunction_read) record id %lld not found\n", gf->rid);
        exit(-1);
    }
    npts = rp->npts;
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
    Record *rp;
    rp = Record_get_pointer(rs, gs->n_record, gf->rid);
    npts = rp->npts;
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

#define _GF_COPY_MAT(var) \
    cudaMemcpy(gbuf.##var, gf_cpu->##var, npts *nfreq * sizeof(Float64), cudaMemcpyHostToDevice)

void GreenFunction_copyto_gpu(GreenFunction *gf_cpu, GreenFunction *gf_gpu, Int64 npts, Int64 nfreq)
{
    GreenFunction gbuf;
    cudaMemcpy(&gbuf, gf_gpu, sizeof(GreenFunction), cudaMemcpyDeviceToHost);
    gbuf.rid = gf_cpu->rid;
    gbuf.eid = gf_cpu->eid;
    _GF_COPY_MAT(g11);
    _GF_COPY_MAT(g22);
    _GF_COPY_MAT(g33);
    _GF_COPY_MAT(g12);
    _GF_COPY_MAT(g13);
    _GF_COPY_MAT(g23);
    cudaDeviceSynchronize();
}

#undef _GF_COPY_MAT

void GreenFunction_copy_gpu_to_cpu(GreenFunction *gf_cpu, GreenFunction *gf_gpu, Int64 npts, Int64 nfreq)
{
    Int64 n;
    GreenFunction gbuf;
    cudaMemcpy(&gbuf, gf_gpu, sizeof(GreenFunction), cudaMemcpyDeviceToHost);

    n = npts * nfreq;
    gf_cpu->rid = gbuf.rid;
    gf_cpu->eid = gbuf.eid;
    cudaMemcpy(gf_cpu->g11, gbuf.g11, n * sizeof(Float64), cudaMemcpyDeviceToHost);
    cudaMemcpy(gf_cpu->g22, gbuf.g22, n * sizeof(Float64), cudaMemcpyDeviceToHost);
    cudaMemcpy(gf_cpu->g33, gbuf.g33, n * sizeof(Float64), cudaMemcpyDeviceToHost);
    cudaMemcpy(gf_cpu->g12, gbuf.g12, n * sizeof(Float64), cudaMemcpyDeviceToHost);
    cudaMemcpy(gf_cpu->g13, gbuf.g13, n * sizeof(Float64), cudaMemcpyDeviceToHost);
    cudaMemcpy(gf_cpu->g23, gbuf.g23, n * sizeof(Float64), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

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
    cudaMalloc(&(gflist->gpu), n * sizeof(GreenFunction));
    // gflist->gpu = NULL;
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
    Int64 i, ndat;
    Record *rp;
    GreenFunction gbuf;
    for (i = 0; i < gflist->n; i++)
    {
#ifdef DEBUG
        printf("(GreenFunction_xPU_read) Reading function %lld\n", i);
#endif
        GreenFunction_read(&(gflist->cpu[i]), rlist->cpu, gs->cpu, fp);
        rp = Record_get_pointer(rlist->cpu, gs->cpu->n_record, gflist->cpu[i].rid);
        ndat = rp->npts * gs->cpu->n_frequency_pair;
        gbuf.rid = gflist->cpu[i].rid;
        gbuf.eid = gflist->cpu[i].eid;
        cudaMalloc(&(gbuf.g11), ndat * sizeof(Float64));
        cudaMalloc(&(gbuf.g22), ndat * sizeof(Float64));
        cudaMalloc(&(gbuf.g33), ndat * sizeof(Float64));
        cudaMalloc(&(gbuf.g12), ndat * sizeof(Float64));
        cudaMalloc(&(gbuf.g13), ndat * sizeof(Float64));
        cudaMalloc(&(gbuf.g23), ndat * sizeof(Float64));
    }
    gflist->mcpu = 1;
    gflist->mgpu = 0;
#ifdef DEBUG
    printf("(GreenFunction_xPU_read) Sync\n");
#endif
    GreenFunction_xPU_sync(gflist, rlist, gs);
}

void GreenFunction_xPU_write(GreenFunction_xPU *gflist, Record_xPU *rlist, GlobalSetting_xPU *gs, FILE *fp)
{
    Int64 i;
    for (i = 0; i < gflist->n; ++i)
        GreenFunction_write(&gflist->cpu[i], rlist->cpu, gs->cpu, fp);
}

void GreenFunction_xPU_sync(GreenFunction_xPU *gflist, Record_xPU *rlist, GlobalSetting_xPU *gs)
{
    Int64 i, ir, npts, nfreq;
#ifdef DEBUG
    printf("(GreenFunction_xPU_sync) start\n");
#endif
    nfreq = gs->cpu->n_frequency_pair;
#ifdef DEBUG
    printf("(GreenFunction_xPU_sync) nfreq: %lld\n", nfreq);
#endif
    if (gflist->mcpu > gflist->mgpu)
    {
#ifdef DEBUG
        printf("(GreenFunction_xPU_sync) CPU -> GPU\n");
#endif
        for (i = 0; i < gflist->n; i++)
        {
#ifdef DEBUG
            printf("(GreenFunction_xPU_sync) gf[%lld]\n", i);
#endif
            for (ir = 0; ir < rlist->n_records; ir++)
                if ((gflist->cpu)[i].rid == rlist->cpu[ir].id)
                    break;
            npts = rlist->cpu[ir].npts;
#ifdef DEBUG
            printf("(GreenFunction_xPU_sync) found record id: %lld, npts: %lld\n", ir, npts);
#endif
            GreenFunction_copyto_gpu(&((gflist->cpu)[i]), &((gflist->gpu)[i]), npts, nfreq);
        }
    }
    if (gflist->mcpu < gflist->mgpu)
    {
#ifdef DEBUG
        printf("(GreenFunction_xPU_sync) GPU -> CPU\n");
#endif
        for (i = 0; i < gflist->n; i++)
        {
            for (ir = 0; ir < rlist->n_records; ir++)
                if (gflist->cpu[i].rid == rlist->cpu[ir].id)
                    break;
            npts = rlist->cpu[ir].npts;
            GreenFunction_copy_gpu_to_cpu(&gflist->cpu[i], &gflist->gpu[i], npts, nfreq);
        }
    }
    gflist->mcpu = 0;
    gflist->mgpu = 0;

#ifdef DEBUG
    printf("(GreenFunction_xPU_sync) Wait for device synchronization\n");
#endif
    cudaDeviceSynchronize();
}
