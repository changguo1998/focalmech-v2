#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "Record.h"

// #define DEBUG

static inline Int64 _max_(Int64 a, Int64 b) { return (a > b) ? a : b; }

void Record_init(Record *r)
{
    r->id = 0;
    r->npts = 0;
    r->data = NULL;
}

void Record_free(Record *r)
{
    if (r->data)
    {
        free(r->data);
        r->data = NULL;
    }
}

void Record_read(Record *r, Int64 nfreq, FILE *fp)
{
#ifdef DEBUG
    printf("(Record_read) start\n");
#endif
    fread(&r->id, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(Record_read) id: %lld\n", r->id);
#endif
    fread(&r->npts, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(Record_read) npts: %lld, nfreq: %lld\n", r->npts, nfreq);
#endif
    Int64 ndat = r->npts * nfreq;
    r->data = (Float64 *)malloc(sizeof(Float64) * ndat);
    fread(r->data, sizeof(Float64), ndat, fp);
}

void Record_write(Record *r, Int64 nfreq, FILE *fp)
{
    fwrite(&r->id, sizeof(Int64), 1, fp);
    fwrite(&r->npts, sizeof(Int64), 1, fp);
    Int64 ndat = r->npts * nfreq;
    fwrite(r->data, sizeof(Float64), ndat, fp);
}

void Record_xPU_alloc(Record_xPU *rs, Int64 n)
{
    rs->mcpu = 0;
    rs->mgpu = 0;
    if (n <= 0)
    {
        rs->n_records = 0;
        rs->cpu = NULL;
        rs->gpu = NULL;
        return;
    }
    rs->cpu = (Record *)malloc(n * sizeof(Record));
    cudaMalloc(&rs->gpu, n * sizeof(Record));
    rs->n_records = n;
}

void Record_xPU_free(Record_xPU *rs)
{
    Int64 i;
    Record rbuf;
    for (i = 0; i < rs->n_records; i++)
    {
        Record_free(&(rs->cpu[i]));
        cudaMemcpy(&rbuf, &(rs->gpu[i]), sizeof(Record), cudaMemcpyDeviceToHost);
        cudaFree(rbuf.data);
    }
    free(rs->cpu);
    cudaFree(rs->gpu);
    rs->cpu = NULL;
    rs->gpu = NULL;
    rs->mcpu = 0;
    rs->mgpu = 0;
    rs->n_records = 0;
}

void Record_xPU_read(Record_xPU *rs, Int64 nfreq, FILE *fp)
{
    Int64 i;
    for (i = 0; i < rs->n_records; i++)
    {
        Record_init(&((rs->cpu)[i]));
        Record_read(&((rs->cpu)[i]), nfreq, fp);
    }
    rs->mgpu = 0;
    rs->mcpu = 1;
#ifdef DEBUG
    printf("(Record_xPU_rea) Sync...\n");
#endif
    Record_xPU_sync(rs, nfreq);
}

void Record_xPU_write(Record_xPU *rs, Int64 nfreq, FILE *fp)
{
    Int64 i;
    Record_xPU_sync(rs, nfreq);
    for (i = 0; i < rs->n_records; i++)
        Record_write(&((rs->cpu)[i]), nfreq, fp);
}

void Record_xPU_sync(Record_xPU *rs, Int64 nfreq)
{
    Int64 i, ndat;
    Record rbuf;
    if (rs->mcpu > rs->mgpu)
        for (i = 0; i < rs->n_records; i++)
        {
            ndat = rs->cpu[i].npts * nfreq;
            cudaMemcpy(&rbuf, &((rs->gpu)[i]), sizeof(Record), cudaMemcpyDeviceToHost);
            if (rbuf.data == NULL)
                cudaMalloc(&rbuf.data, ndat * sizeof(Float64));
            cudaMemcpy(rbuf.data, rs->cpu[i].data, ndat * sizeof(Float64), cudaMemcpyHostToDevice);
            rbuf.id = rs->cpu[i].id;
            rbuf.npts = rs->cpu[i].npts;
            cudaMemcpy(&(rs->gpu[i]), &rbuf, sizeof(Record), cudaMemcpyHostToDevice);
        }
    if (rs->mcpu < rs->mgpu)
        for (i = 0; i < rs->n_records; i++)
        {
            cudaMemcpy(&rbuf, &((rs->gpu)[i]), sizeof(Record), cudaMemcpyDeviceToHost);
            ndat = rbuf.npts * nfreq;
            cudaMemcpy(rs->cpu[i].data, rbuf.data, ndat * sizeof(Float64), cudaMemcpyDeviceToHost);
            rs->cpu[i].id = rbuf.id;
            rs->cpu[i].npts = rbuf.npts;
        }
    rs->mcpu = 0;
    rs->mgpu = 0;
    cudaDeviceSynchronize();
}
