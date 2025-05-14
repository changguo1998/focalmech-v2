#ifndef __GLOBALSETTING_H__
#define __GLOBALSETTING_H__

#include <stdio.h>
#include "types.h"

#define MAX_INVERSION_TAG_LENGTH 128

typedef struct _GlobalSetting_t
{
    char tag[MAX_INVERSION_TAG_LENGTH];
    Int64 n_record, n_event_location, n_frequency_pair, n_phase;
    Int64 nstrike, ndip, nrake;
    Float64 dstrike, ddip, drake;
} GlobalSetting;

void GlobalSetting_read(GlobalSetting *s, FILE *fp);
void GlobalSetting_write(GlobalSetting *s, FILE *fp);

typedef struct _GlobalSetting_xPU_t
{
    Int64 mcpu, mgpu;
    GlobalSetting *cpu, *gpu;
} GlobalSetting_xPU;

void GlobalSetting_xPU_alloc(GlobalSetting_xPU *gs);
void GlobalSetting_xPU_free(GlobalSetting_xPU *gs);
void GlobalSetting_xPU_read(GlobalSetting_xPU *gs, FILE *fp);
void GlobalSetting_xPU_write(GlobalSetting_xPU *gs, FILE *fp);
void GlobalSetting_xPU_sync(GlobalSetting_xPU *gs);

#endif // __GLOBALSETTING_H__
