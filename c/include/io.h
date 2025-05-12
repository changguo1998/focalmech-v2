#ifndef __IO_H__
#define __IO_H__

#include "types.h"
#include "GlobalSetting.h"
#include "Record.h"
#include "GreenFunction.h"
#include "Phase.h"

void load_database(GlobalSetting_xPU *gs, Record_xPU *rlist, GreenFunction_xPU *gflist, Phase_xPU *plist, FILE *fp);
void save_database(GlobalSetting_xPU *gs, Record_xPU *rs, GreenFunction_xPU *gfs, Phase_xPU *plist, FILE *fp);

#endif // __IO_H__
