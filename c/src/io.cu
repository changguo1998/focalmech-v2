#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "GlobalSetting.h"
#include "Record.h"
#include "GreenFunction.h"
#include "Phase.h"
#include "io.h"

#define DEBUG

void load_database(GlobalSetting_xPU *gs, Record_xPU *rlist, GreenFunction_xPU *gflist, Phase_xPU *plist, FILE *fp)
{
    Int64 i;
#ifdef DEBUG
    printf("(load_database) allocate GlobalSetting\n");
#endif
    GlobalSetting_xPU_alloc(gs);
#ifdef DEBUG
    printf("(load_database) load GlobalSetting\n");
#endif
    GlobalSetting_xPU_read(gs, fp);
#ifdef DEBUG
    printf("(load_database) allocate records list\n");
#endif
    Record_xPU_alloc(rlist, gs->cpu->n_record);
#ifdef DEBUG
    printf("(load_database) read each record\n");
#endif
    Record_xPU_read(rlist, gs->cpu->n_frequency_pair, fp);

#ifdef DEBUG
    printf("(load_database) allocate gf list\n");
#endif
    GreenFunction_xPU_alloc(gflist, gs->cpu->n_event_location * gs->cpu->n_record);

#ifdef DEBUG
    printf("(load_database) read gf\n");
#endif
    GreenFunction_xPU_read(gflist, rlist, gs, fp);

#ifdef DEBUG
    printf("(load_database) allocate phase list\n");
#endif
    Phase_xPU_alloc(plist, gs->cpu->n_phase);
#ifdef DEBUG
    printf("(load_database) read phase list\n");
#endif
    Phase_xPU_read(plist, fp);
}

void save_database(GlobalSetting_xPU *gs, Record_xPU *rs, GreenFunction_xPU *gfs, Phase_xPU *plist, FILE *fp)
{
    Int64 i;
    GlobalSetting_xPU_write(gs, fp);
    Record_xPU_write(rs, gs->cpu->n_frequency_pair, fp);
    GreenFunction_xPU_write(gfs, rs, gs, fp);
    Phase_xPU_write(plist, fp);
}
