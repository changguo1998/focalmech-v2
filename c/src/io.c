#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "io.h"

// #define DEBUG

void load_database(GlobalSetting *gs, Record **rlist, GreenFun **gflist, FILE *fp)
{
    Int64 i;
#ifdef DEBUG
    printf("(load_database) load GlobalSetting\n");
#endif
    GlobalSetting_read(gs, fp);
#ifdef DEBUG
    printf("(load_database) allocate records list\n");
#endif
    *rlist = (Record *)malloc((gs->n_record) * sizeof(Record));
#ifdef DEBUG
    printf("(load_database) read each record\n");
#endif
    for (i = 0; i < gs->n_record; i++)
    {
        Record_init(&((*rlist)[i]));
        Record_read(&((*rlist)[i]), gs, fp);
    }

#ifdef DEBUG
    printf("(load_database) allocate gf list\n");
#endif
    *gflist = (GreenFun *)malloc((gs->n_event_location * gs->n_record) * sizeof(GreenFun));
    for (i = 0; i < gs->n_event_location * gs->n_record; i++)
        GreenFun_read(&((*gflist)[i]), *rlist, gs, fp);
}

void save_database(GlobalSetting *gs, Record *rs, GreenFun *gfs, FILE *fp)
{
    Int64 i;
    GlobalSetting_write(gs, fp);
    for (i = 0; i < gs->n_record; i++)
        Record_write(&rs[i], gs, fp);
    for (i = 0; i < gs->n_event_location * gs->n_record; i++)
        GreenFun_write(&gfs[i], rs, gs, fp);
}
