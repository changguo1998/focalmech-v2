#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "types.h"
#include "GlobalSetting.h"
#include "Record.h"
#include "GreenFunction.h"
#include "Phase.h"
#include "Result.h"
#include "misfits.h"
#include "kernel.h"
#include "io.h"

// #define GPU
#define DEBUG

#define WRITE_LOG(...)                \
    do                                \
    {                                 \
        printf(__VA_ARGS__);          \
        fprintf(fp_log, __VA_ARGS__); \
    } while (0)

static inline Int64 _max_(Int64 a, Int64 b) { return (a > b) ? a : b; }

int main(int argc, char *argv[])
{
    FILE *fp_input, *fp_log, *fp_output;
    Int64 n_freq, n_phase, n_fm;

    GlobalSetting_xPU global_setting;
    Record_xPU record;
    GreenFunction_xPU greenfunction_database;
    Phase_xPU phase_list;
    Result_xPU result_buffer;

    fp_input = fopen("input_db.bin", "rb");
    if (fp_input == NULL)
    {
        printf("Error opening file input\n");
        exit(1);
    }

    fp_log = fopen("log.txt", "w");
    if (fp_log == NULL)
    {
        printf("Error opening file log\n");
        exit(-1);
    }

    WRITE_LOG("Load input database\n");
    load_database(&global_setting, &record, &greenfunction_database, &phase_list, fp_input);
    fclose(fp_input);

    WRITE_LOG("Allocate result buffers\n");
    n_freq = global_setting.cpu->n_frequency_pair;
    n_phase = global_setting.cpu->n_phase;
    n_fm = global_setting.cpu->nstrike * global_setting.cpu->ndip * global_setting.cpu->nrake;
#ifdef DEBUG
    printf("n_freq: %lld, n_phase: %lld, nfm :% lld\n", n_freq, n_phase, n_fm);
#endif
    Result_xPU_alloc(&result_buffer, n_freq, n_phase, n_fm);
    Int64 nresult = n_freq * n_phase * n_fm;
    memset(result_buffer.waveform, 0, nresult * sizeof(Float64));
    memset(result_buffer.shift, 0, nresult * sizeof(Int64));
    memset(result_buffer.polarity, 0, nresult * sizeof(Float64));
    memset(result_buffer.ps_ratio, 0, nresult * sizeof(Float64));
    result_buffer.mcpu = 1;
    result_buffer.mgpu = 0;
    Result_xPU_sync(&result_buffer);
    Int64 i;
    for (i = 0; i < nresult; i++)
    {
        result_buffer.waveform[i] = 10.0;
        result_buffer.shift[i] = 10;
        result_buffer.polarity[i] = 10.0;
        result_buffer.ps_ratio[i] = 10.0;
    }

    WRITE_LOG("Start kernel\n");
#ifdef GPU
    call_kernel_gpu(&global_setting, &record, &greenfunction_database, &phase_list, &result_buffer);
#else
    call_kernel_omp(&global_setting, &record, &greenfunction_database, &phase_list, &result_buffer);
#endif

    WRITE_LOG("Write result to file\n");
#ifdef GPU
    fp_output = fopen("result_gpu.bin", "wb");
#else
    fp_output = fopen("result_omp.bin", "wb");
#endif
    if (fp_output == NULL)
    {
        printf("Error opening output\n");
        exit(-1);
    }
    Result_xPU_write(&result_buffer, fp_output);
    fclose(fp_output);

    WRITE_LOG("Free memory\n");
    Result_xPU_free(&result_buffer);
    Phase_xPU_free(&phase_list);
    GreenFunction_xPU_free(&greenfunction_database);
    Record_xPU_free(&record);
    GlobalSetting_xPU_free(&global_setting);
    WRITE_LOG("End of program\n");
    fclose(fp_log);
    printf("Done\n");

    return 0;
}
