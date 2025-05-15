#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "types.h"
#include "GlobalSetting.h"
#include "Record.h"
#include "GreenFunction.h"
#include "Phase.h"
#include "io.h"

int main()
{
    Int64 i;
    FILE *fp;
    GlobalSetting_xPU gs;
    Record_xPU rlist;
    GreenFunction_xPU gflist;
    Phase_xPU plist;

    Record *r;
    GreenFunction *gf;
    Phase *p;

    printf("open file\n");
    fp = fopen("input_db.bin", "rb");
    printf("load data\n");
    load_database(&gs, &rlist, &gflist, &plist, fp);
    fclose(fp);
    printf("global setting:\n");
    printf("\ttag: %s\n", gs.cpu->tag);
    printf("\tn record: %lld\n", gs.cpu->n_record);
    printf("\tn event location: %lld\n", gs.cpu->n_event_location);
    printf("\tn frequency pairs: %lld\n", gs.cpu->n_frequency_pair);
    printf("\tn phases: %lld\n", gs.cpu->n_phase);
    printf("\tnstrike: %lld \tndip: %lld, \tnrake: %lld\n\n", gs.cpu->nstrike, gs.cpu->ndip, gs.cpu->nrake);
    printf("\tdstrike: %lf \tddip: %lf, \tdrake: %lf\n\n", gs.cpu->dstrike, gs.cpu->ddip, gs.cpu->drake);

    printf("records:\n");
    for (i = 0; i < gs.cpu->n_record; i++)
    {
        r = &rlist.cpu[i];
        printf("\tid: %lld, npts: %lld\n", r->id, r->npts);
    }

    printf("green functions:\n");
    for (i = 0; i < gs.cpu->n_record * gs.cpu->n_event_location; i++)
    {
        gf = &gflist.cpu[i];
        printf("\trid: %lld, eid: %lld\n", gf->rid, gf->eid);
    }

    printf("phases:\n");
    for (i = 0; i < gs.cpu->n_phase; i++)
    {
        p = &plist.cpu[i];
        printf("\trid: %lld, eid: %lld, type: %lld, Rstart: %lld, Estart: %lld, length: %lld, flag: %d\n",
               p->rid, p->eid, p->type, p->Rstart, p->Estart, p->length, (int)p->flag);
    }

    fp = fopen("output.bin", "wb");
    save_database(&gs, &rlist, &gflist, &plist, fp);
    fclose(fp);

    GlobalSetting_xPU_free(&gs);
    Record_xPU_free(&rlist);
    GreenFunction_xPU_free(&gflist);
    Phase_xPU_free(&plist);

    return 0;
}
