#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "Phase.h"

void Phase_read(Phase *p, FILE *fp)
{
    fread(&p->rid, sizeof(Int64), 1, fp);
    fread(&p->eid, sizeof(Int64), 1, fp);
    fread(&p->type, sizeof(Int64), 1, fp);
    fread(&p->Rstart, sizeof(Int64), 1, fp);
    fread(&p->Estart, sizeof(Int64), 1, fp);
    fread(&p->length, sizeof(Int64), 1, fp);
    fread(&p->flag, sizeof(UInt8), 1, fp);
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
    ps->mcpu = ps->mgpu + 1;
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
