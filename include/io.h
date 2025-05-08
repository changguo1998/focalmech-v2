#ifndef __IO_H__
#define __IO_H__

#include "types.h"

void load_database(GlobalSetting *gs, Record **rlist, GreenFun **gflist, FILE *fp);
void save_database(GlobalSetting *gs, Record *rs, GreenFun *gfs, FILE *fp);

#endif // __IO_H__
