#ifndef __TYPES_H__
#define __TYPES_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_INVERSION_TAG_LENGTH 128

typedef unsigned long long UInt64;
typedef long long Int64;
typedef unsigned char UInt8;
typedef float Float32;
typedef double Float64;

typedef struct _GlobalSetting_t
{
    char tag[MAX_INVERSION_TAG_LENGTH];
    Int64 n_record, n_event_location, n_frequency_pairs;
    Float64 dstrike, ddip, drake;
} GlobalSetting;

void GlobalSetting_read(GlobalSetting *s, FILE *fp);
void GlobalSetting_write(GlobalSetting *s, FILE *fp);

typedef struct _Phase_t
{
    Int64 type, time;
    UInt8 flag;
} Phase;

void Phase_read(Phase *p, FILE *fp);
void Phase_write(Phase *p, FILE *fp);

typedef struct _Record_t
{
    Int64 id, npts, nphase;
    Float64 *data;
    Phase *phase;
} Record;

void Record_init(Record *r);
void Record_free(Record *r);
void Record_read(Record *r, GlobalSetting *gs, FILE *fp);
void Record_write(Record *r, GlobalSetting *gs, FILE *fp);

typedef struct _GreenFun_t
{
    Int64 rid, eid;
    Float64 *g11, *g22, *g33, *g12, *g13, *g23;
} GreenFun;

void GreenFun_init(GreenFun *gf);
void GreenFun_free(GreenFun *gf);
void GreenFun_read(GreenFun *gf, Record *rs, GlobalSetting *gs, FILE *fp);
void GreenFun_write(GreenFun *gf, Record *rs, GlobalSetting *gs, FILE *fp);

#endif // __TYPES_H__
