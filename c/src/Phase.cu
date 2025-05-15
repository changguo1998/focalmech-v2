#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "Phase.h"

// #define DEBUG

static inline Int64 _max_(Int64 a, Int64 b) { return (a > b) ? a : b; }

void Phase_read(Phase *p, FILE *fp)
{
#ifdef DEBUG
    printf("(Phase_read) start\n");
#endif
    fread(&p->rid, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(Phase_read) record id: %lld\n", p->rid);
#endif
    fread(&p->eid, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(Phase_read) event id: %lld\n", p->eid);
#endif
    fread(&p->type, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(Phase_read) type: %lld\n", p->type);
#endif
    fread(&p->Rstart, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(Phase_read) Rstart: %lld\n", p->Rstart);
#endif
    fread(&p->Estart, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(Phase_read) Estart: %lld\n", p->Estart);
#endif
    fread(&p->length, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(Phase_read) length: %lld\n", p->length);
#endif
    fread(&p->flag, sizeof(UInt8), 1, fp);
#ifdef DEBUG
    printf("(Phase_read) flag: %d\n", p->flag);
#endif
}

void Phase_write(Phase *p, FILE *fp)
{
    fwrite(&p->rid, sizeof(Int64), 1, fp);
    fwrite(&p->eid, sizeof(Int64), 1, fp);
    fwrite(&p->type, sizeof(Int64), 1, fp);
    fwrite(&p->Rstart, sizeof(Int64), 1, fp);
    fwrite(&p->Estart, sizeof(Int64), 1, fp);
    fwrite(&p->length, sizeof(Int64), 1, fp);
    fwrite(&p->flag, sizeof(UInt8), 1, fp);
}

void Phase_xPU_alloc(Phase_xPU *ps, Int64 n)
{
    ps->mcpu = 0;
    ps->mgpu = 0;
    if (n <= 0)
    {
        ps->cpu = NULL;
        ps->gpu = NULL;
        ps->nphases = 0;
        return;
    }
    ps->cpu = (Phase *)malloc(n * sizeof(Phase));
    cudaMalloc(&(ps->gpu), n * sizeof(Phase));
    ps->nphases = n;
}

void Phase_xPU_free(Phase_xPU *ps)
{
    if (ps->cpu)
    {
        free(ps->cpu);
        ps->cpu = NULL;
    }
    if (ps->gpu)
    {
        cudaFree(ps->gpu);
        ps->gpu = NULL;
    }
    ps->mcpu = 0;
    ps->mgpu = 0;
    ps->nphases = 0;
}

void Phase_xPU_read(Phase_xPU *ps, FILE *fp)
{
    Int64 i;
    for (i = 0; i < ps->nphases; i++)
        Phase_read(&(ps->cpu[i]), fp);
    ps->mcpu = 1;
    ps->mgpu = 0;
    Phase_xPU_sync(ps);
}

void Phase_xPU_write(Phase_xPU *ps, FILE *fp)
{
    Int64 i;
    Phase_xPU_sync(ps);
    for (i = 0; i < ps->nphases; i++)
        Phase_write(&(ps->cpu[i]), fp);
}

void Phase_xPU_sync(Phase_xPU *ps)
{
    if (ps->mcpu > ps->mgpu)
        cudaMemcpy(ps->gpu, ps->cpu, ps->nphases * sizeof(Phase), cudaMemcpyHostToDevice);
    if (ps->mcpu < ps->mgpu)
        cudaMemcpy(ps->cpu, ps->gpu, ps->nphases * sizeof(Phase), cudaMemcpyDeviceToHost);
    ps->mcpu = 0;
    ps->mcpu = 0;
    cudaDeviceSynchronize();
}
