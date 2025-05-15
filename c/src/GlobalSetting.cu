#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "GlobalSetting.h"

// #define DEBUG

static inline Int64 _max_(Int64 a, Int64 b) { return (a > b) ? a : b; }

void GlobalSetting_read(GlobalSetting *s, FILE *fp)
{
#ifdef DEBUG
    printf("(GlobalSetting_read) start\n");
#endif
    fread(s->tag, sizeof(char), MAX_INVERSION_TAG_LENGTH, fp);
#ifdef DEBUG
    printf("(GlobalSetting_read) tag: %s\n", s->tag);
#endif
    fread(&s->n_record, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(GlobalSetting_read) n_record: %lld\n", s->n_record);
#endif
    fread(&s->n_event_location, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(GlobalSetting_read) n_event_location: %lld\n", s->n_event_location);
#endif
    fread(&s->n_frequency_pair, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(GlobalSetting_read) n_freq: %lld\n", s->n_frequency_pair);
#endif
    fread(&s->n_phase, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(GlobalSetting_read) n_phase: %lld\n", s->n_phase);
#endif
    fread(&s->nstrike, sizeof(Int64), 1, fp);
    fread(&s->ndip, sizeof(Int64), 1, fp);
    fread(&s->nrake, sizeof(Int64), 1, fp);
    fread(&s->dstrike, sizeof(Float64), 1, fp);
#ifdef DEBUG
    printf("(GlobalSetting_read) dstrike: %lf\n", s->dstrike);
#endif
    fread(&s->ddip, sizeof(Float64), 1, fp);
#ifdef DEBUG
    printf("(GlobalSetting_read) ddip: %lf\n", s->ddip);
#endif
    fread(&s->drake, sizeof(Float64), 1, fp);
#ifdef DEBUG
    printf("(GlobalSetting_read) drake: %lf\n", s->drake);
#endif
}

void GlobalSetting_write(GlobalSetting *s, FILE *fp)
{
    fwrite(s->tag, sizeof(char), MAX_INVERSION_TAG_LENGTH, fp);
    fwrite(&s->n_record, sizeof(Int64), 1, fp);
    fwrite(&s->n_event_location, sizeof(Int64), 1, fp);
    fwrite(&s->n_frequency_pair, sizeof(Int64), 1, fp);
    fwrite(&s->n_phase, sizeof(Int64), 1, fp);
    fwrite(&s->nstrike, sizeof(Int64), 1, fp);
    fwrite(&s->ndip, sizeof(Int64), 1, fp);
    fwrite(&s->nrake, sizeof(Int64), 1, fp);
    fwrite(&s->dstrike, sizeof(Float64), 1, fp);
    fwrite(&s->ddip, sizeof(Float64), 1, fp);
    fwrite(&s->drake, sizeof(Float64), 1, fp);
}

void GlobalSetting_xPU_alloc(GlobalSetting_xPU *gs)
{
    gs->cpu = (GlobalSetting *)malloc(sizeof(GlobalSetting));
    cudaMalloc(&(gs->gpu), sizeof(GlobalSetting));
    gs->mcpu = 0;
    gs->mgpu = 0;
}

void GlobalSetting_xPU_free(GlobalSetting_xPU *gs)
{
    if (gs->cpu)
    {
        free(gs->cpu);
        gs->cpu = NULL;
    }
    if (gs->gpu)
    {
        cudaFree(gs->gpu);
        gs->gpu = NULL;
    }
    gs->mcpu = 0;
    gs->mgpu = 0;
}

void GlobalSetting_xPU_read(GlobalSetting_xPU *gs, FILE *fp)
{
    GlobalSetting_read(gs->cpu, fp);
    gs->mcpu = 1;
    gs->mgpu = 0;
    GlobalSetting_xPU_sync(gs);
}

void GlobalSetting_xPU_write(GlobalSetting_xPU *gs, FILE *fp)
{
    GlobalSetting_xPU_sync(gs);
    GlobalSetting_write(gs->cpu, fp);
}

void GlobalSetting_xPU_sync(GlobalSetting_xPU *gs)
{
    if (gs->mcpu > gs->mgpu)
        cudaMemcpy(gs->gpu, gs->cpu, sizeof(GlobalSetting), cudaMemcpyHostToDevice);

    if (gs->mcpu < gs->mgpu)
        cudaMemcpy(gs->cpu, gs->gpu, sizeof(GlobalSetting), cudaMemcpyDeviceToHost);

    gs->mcpu = 0;
    gs->mgpu = 0;
    cudaDeviceSynchronize();
}
