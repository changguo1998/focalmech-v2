#ifndef __PHASE_H__
#define __PHASE_H__

#include "types.h"

#include "GlobalSetting.h"
#include "Record.h"

typedef struct _Phase_t
{
    Int64 rid, eid, type, Rstart, Estart, length;
    UInt8 flag;
} Phase;

void Phase_read(Phase *p, FILE *fp);
void Phase_write(Phase *p, FILE *fp);

typedef struct _Phase_xPU_t
{
    Int64 mcpu, mgpu, nphases;
    Phase *cpu, *gpu;
} Phase_xPU;

void Phase_xPU_alloc(Phase_xPU *ps, Int64 n);
void Phase_xPU_free(Phase_xPU *ps);
void Phase_xPU_read(Phase_xPU *ps, FILE *fp);
void Phase_xPU_write(Phase_xPU *ps, FILE *fp);
void Phase_xPU_sync(Phase_xPU *ps);

#endif // __PHASE_H__
