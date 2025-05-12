#include <string.h>

#include "types.h"

// #define DEBUG

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
    fread(&s->n_frequency_pairs, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(GlobalSetting_read) n_freq: %lld\n", s->n_frequency_pairs);
#endif
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
    fwrite(&s->n_frequency_pairs, sizeof(Int64), 1, fp);
    fwrite(&s->dstrike, sizeof(Float64), 1, fp);
    fwrite(&s->ddip, sizeof(Float64), 1, fp);
    fwrite(&s->drake, sizeof(Float64), 1, fp);
}

void Phase_read(Phase *p, FILE *fp)
{
    fread(&p->type, sizeof(Int64), 1, fp);
    fread(&p->time, sizeof(Int64), 1, fp);
    fread(&p->flag, sizeof(UInt8), 1, fp);
}

void Phase_write(Phase *p, FILE *fp)
{
    fwrite(&p->type, sizeof(Int64), 1, fp);
    fwrite(&p->time, sizeof(Int64), 1, fp);
    fwrite(&p->flag, sizeof(UInt8), 1, fp);
}

void Record_init(Record *r)
{
    r->id = 0;
    r->npts = 0;
    r->data = NULL;
    r->nphase = 0;
    r->phase = NULL;
}

void Record_free(Record *r)
{
    if (r->data)
    {
        free(r->data);
        r->data = NULL;
    }
    if (r->phase)
    {
        free(r->phase);
        r->phase = NULL;
    }
}

void Record_read(Record *r, GlobalSetting *gs, FILE *fp)
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
    printf("(Record_read) npts: %lld\n", r->npts);
#endif
    fread(&r->nphase, sizeof(Int64), 1, fp);
#ifdef DEBUG
    printf("(Record_read) nphase: %lld\n", r->nphase);
#endif
    Int64 ndat = r->npts * gs->n_frequency_pairs;
    r->data = (Float64 *)malloc(sizeof(Float64) * ndat);
    fread(r->data, sizeof(Float64), ndat, fp);
    r->phase = (Phase *)malloc(sizeof(Phase) * r->nphase);
    for (Int64 i = 0; i < r->nphase; i++)
        Phase_read(&(r->phase[i]), fp);
}

void Record_write(Record *r, GlobalSetting *gs, FILE *fp)
{
    fwrite(&r->id, sizeof(Int64), 1, fp);
    fwrite(&r->npts, sizeof(Int64), 1, fp);
    fwrite(&r->nphase, sizeof(Int64), 1, fp);
    Int64 ndat = r->npts * gs->n_frequency_pairs;
    fwrite(r->data, sizeof(Float64), ndat, fp);
    for (Int64 i = 0; i < r->nphase; i++)
        Phase_write(&r->phase[i], fp);
}

void GreenFun_init(GreenFun *gf)
{
    gf->rid = 0;
    gf->eid = 0;
    gf->g11 = NULL;
    gf->g22 = NULL;
    gf->g33 = NULL;
    gf->g12 = NULL;
    gf->g13 = NULL;
    gf->g23 = NULL;
}

#define FREE_MEMORY(var) \
    if (var)             \
        do               \
        {                \
            free(var);   \
            var = NULL;  \
    } while (0)

void GreenFun_free(GreenFun *gf)
{
    FREE_MEMORY(gf->g11);
    FREE_MEMORY(gf->g22);
    FREE_MEMORY(gf->g33);
    FREE_MEMORY(gf->g12);
    FREE_MEMORY(gf->g13);
    FREE_MEMORY(gf->g23);
}

#undef FREE_MEMORY

#define GF_READ_MAT(var)                             \
    var = (Float64 *)malloc(ndat * sizeof(Float64)); \
    fread(var, sizeof(Float64), ndat, fp)

void GreenFun_read(GreenFun *gf, Record *rs, GlobalSetting *gs, FILE *fp)
{
    fread(&gf->rid, sizeof(Int64), 1, fp);
    fread(&gf->eid, sizeof(Int64), 1, fp);
    Int64 i, npts;
    for (i = 0; i < gs->n_record; i++)
    {
        if (rs[i].id == gf->rid)
        {
            npts = rs[i].npts;
            break;
        }
    }
    Int64 ndat = npts * gs->n_frequency_pairs;
    GF_READ_MAT(gf->g11);
    GF_READ_MAT(gf->g22);
    GF_READ_MAT(gf->g33);
    GF_READ_MAT(gf->g12);
    GF_READ_MAT(gf->g13);
    GF_READ_MAT(gf->g23);
}

#undef GF_READ_MAT

void GreenFun_write(GreenFun *gf, Record *rs, GlobalSetting *gs, FILE *fp)
{
    Int64 i, npts;
    for (i = 0; i < gs->n_record; i++)
    {
        if (rs[i].id == gf->rid)
        {
            npts = rs[i].npts;
            break;
        }
    }
    Int64 ndat = npts * gs->n_frequency_pairs;
    fwrite(&gf->rid, sizeof(Int64), 1, fp);
    fwrite(&gf->eid, sizeof(Int64), 1, fp);
    fwrite(gf->g11, sizeof(Float64), ndat, fp);
    fwrite(gf->g22, sizeof(Float64), ndat, fp);
    fwrite(gf->g33, sizeof(Float64), ndat, fp);
    fwrite(gf->g12, sizeof(Float64), ndat, fp);
    fwrite(gf->g13, sizeof(Float64), ndat, fp);
    fwrite(gf->g23, sizeof(Float64), ndat, fp);
}
