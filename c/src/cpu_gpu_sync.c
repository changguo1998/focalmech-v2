#include <stdlib.h>
#include <cuda.h>

#include "types.h"
#include "cpu_gpu_sync.h"

void GlobalSetting_copy_cpu_to_gpu(GlobalSetting *gs_cpu, GlobalSetting **gs_gpu)
{
    cudaMalloc(gs_gpu, sizeof(GlobalSetting));
    cudaMemcpy(*gs_gpu, gs_cpu, sizeof(GlobalSetting), cudaMemcpyHostToDevice);
}

void Record_copy_list_cpu_to_gpu(Record *rs_cpu, GlobalSetting *gs_cpu, Record **rs_gpu)
{
    Int64 i, ndata;
    Record rs_gpu_tmp;
    cudaMalloc(rs_gpu, gs_cpu->n_record * sizeof(Record));
    for (i = 0; i < gs_cpu->n_record; i++)
    {
        rs_gpu_tmp.id = rs_cpu[i].id;
        rs_gpu_tmp.npts = rs_cpu[i].npts;
        rs_gpu_tmp.nphase = rs_cpu[i].nphase;
        ndata = rs_cpu[i].npts * gs_cpu->n_frequency_pairs;
        cudaMalloc(&rs_gpu_tmp.data, sizeof(Float64) * ndata);
        cudaMemcpy(rs_gpu_tmp.data, rs_cpu[i].data, ndata * sizeof(Float64), cudaMemcpyHostToDevice);
        cudaMalloc(&rs_gpu_tmp.phase, sizeof(Phase) * rs_cpu[i].nphase);
        cudaMemcpy(rs_gpu_tmp.phase, rs_cpu[i].phase, rs_cpu[i].nphase * sizeof(Phase), cudaMemcpyHostToDevice);
        cudaMemcpy(&((*rs_gpu)[i]), &rs_gpu_tmp, sizeof(Record), cudaMemcpyHostToDevice);
    }
}

#define GF_COPY_G_MAT_TO_GPU(var)                           \
    cudaMalloc(&gf_gpu_tmp.##var, ndata * sizeof(Float64)); \
    cudaMemcpy(gf_gpu_tmp.##var, gfs_cpu[i].##var, ndata * sizeof(Float64), cudaMemcpyDeviceToDevice)

void GreenFun_copy_list_cpu_to_gpu(GreenFun *gfs_cpu, Record *rs_cpu, GlobalSetting *gs_cpu, GreenFun **gfs_gpu)
{
    Int64 i, j, ndata, ngf, npts;
    GreenFun gf_gpu_tmp;
    ngf = gs_cpu->n_event_location * gs_cpu->n_record;
    cudaMalloc(gfs_gpu, ngf * sizeof(GreenFun));
    for (i = 0; i < ngf; i++)
    {
        gf_gpu_tmp.rid = gfs_cpu[i].rid;
        gf_gpu_tmp.eid = gfs_cpu[i].eid;
        for (j = 0; j < gs_cpu->n_record; j++)
        {
            if (rs_cpu[j].id == gf_gpu_tmp.rid)
            {
                npts = rs_cpu[i].npts;
                break;
            }
        }
        ndata = npts * gs_cpu->n_frequency_pairs;
        GF_COPY_G_MAT_TO_GPU(g11);
        GF_COPY_G_MAT_TO_GPU(g22);
        GF_COPY_G_MAT_TO_GPU(g33);
        GF_COPY_G_MAT_TO_GPU(g12);
        GF_COPY_G_MAT_TO_GPU(g13);
        GF_COPY_G_MAT_TO_GPU(g23);
        cudaMemcpy(&((*gfs_gpu)[i]), &gf_gpu_tmp, sizeof(GreenFun), cudaMemcpyHostToDevice);
    }
}

#undef GF_COPY_G_MAT_TO_GPU
