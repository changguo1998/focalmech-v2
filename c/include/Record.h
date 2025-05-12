#ifndef __RECORD_H__
#define __RECORD_H__

#include <stdio.h>
#include "GlobalSetting.h"
#include "Record.h"

typedef struct _Record_t
{
    Int64 id, npts;
    Float64 *data;
} Record;

void Record_init(Record *r);
void Record_free(Record *r);
void Record_read(Record *r, Int64 nfreq, FILE *fp);
void Record_write(Record *r, Int64 nfreq, FILE *fp);

typedef struct _Record_xPU_t
{
    Int64 mcpu, mgpu, n_records;
    Record *cpu, *gpu;
} Record_xPU;

void Record_xPU_alloc(Record_xPU *rs, Int64 n);
void Record_xPU_free(Record_xPU *rs);
void Record_xPU_read(Record_xPU *rs, Int64 nfreq, FILE *fp);
void Record_xPU_write(Record_xPU *rs, Int64 nfreq, FILE *fp);
void Record_xPU_sync(Record_xPU *rs, Int64 nfreq);

#endif // __RECORD_H__
