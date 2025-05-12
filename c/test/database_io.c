#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "types.h"
#include "io.h"

int main()
{
    Int64 i;
    FILE *fp;
    GlobalSetting gs;
    Record *rlist, *r;
    GreenFun *gflist, *gf;

    printf("open file\n");
    fp = fopen("input_db.bin", "rb");
    printf("load data\n");
    load_database(&gs, &rlist, &gflist, fp);
    fclose(fp);
    printf("global setting:\n");
    printf("\ttag: %s\n", gs.tag);
    printf("\tn record: %lld\n", gs.n_record);
    printf("\tn event location: %lld\n", gs.n_event_location);
    printf("\tn frequency pairs: %lld\n", gs.n_frequency_pairs);
    printf("\tdstrike: %lf \tddip: %lf, \tdrake: %lf\n\n", gs.dstrike, gs.ddip, gs.drake);

    printf("records:\n");
    for (i = 0; i < gs.n_record; i++)
    {
        r = &rlist[i];
        printf("\tid: %lld, npts: %lld, nphase: %lld\n\n\n", r->id, r->npts, r->nphase);
    }

    printf("green functions:\n");
    for (i = 0; i < gs.n_record * gs.n_event_location; i++)
    {
        gf = &gflist[i];
        printf("\trid: %lld, eid: %lld\n\n\n", gf->rid, gf->eid);
    }

    fp = fopen("output.bin", "wb");
    save_database(&gs, rlist, gflist, fp);
    fclose(fp);

    for (i = 0; i < gs.n_record; i++)
        Record_free(&rlist[i]);
    free(rlist);

    for (i = 0; i < gs.n_record * gs.n_event_location; i++)
        GreenFun_free(&gflist[i]);
    free(gflist);

    return 0;
}
