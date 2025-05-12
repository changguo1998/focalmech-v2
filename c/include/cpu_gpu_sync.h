#ifndef __CPU_GPU_SYNC_H__
#define __CPU_GPU_SYNC_H__

#include "types.h"

void GlobalSetting_copy_cpu_to_gpu(GlobalSetting *gs_cpu, GlobalSetting **gs_gpu);

void Record_copy_list_cpu_to_gpu(Record *rs_cpu, GlobalSetting *gs_cpu, Record **rs_gpu);

void GreenFun_copy_list_cpu_to_gpu(GreenFun *gfs_cpu, Record *rs_cpu, GlobalSetting *gs_cpu, GreenFun **gfs_gpu);

#endif // __CPU_GPU_SYNC_H__
