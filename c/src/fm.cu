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
    Int64 n_fm;

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

    WRITE_LOG("load input database\n");
    load_database(&global_setting, &record, &greenfunction_database, &phase_list, fp_input);
    fclose(fp_input);

    WRITE_LOG("allocate result buffers\n");
    n_fm = global_setting.cpu->nstrike * global_setting.cpu->ndip * global_setting.cpu->nrake;
    Result_xPU_alloc(&result_buffer, global_setting.cpu->n_phase, n_fm);

    WRITE_LOG("start kernel\n");
    kernel(&global_setting, &record, &greenfunction_database, &phase_list, &result_buffer);
    result_buffer.mgpu = _max_(result_buffer.mcpu, result_buffer.mgpu) + 1;

    WRITE_LOG("write result to file\n");
    fp_output = fopen("result.bin", "wb");
    if (fp_output == NULL)
    {
        printf("Error opening output\n");
        exit(-1);
    }
    Result_xPU_write(&result_buffer, fp_output);
    fclose(fp_output);

    WRITE_LOG("free memory\n");
    Result_xPU_free(&result_buffer);
    Phase_xPU_free(&phase_list);
    GreenFunction_xPU_free(&greenfunction_database);
    Record_xPU_free(&record);
    GlobalSetting_xPU_free(&global_setting);
    WRITE_LOG("end of program\n");
    fclose(fp_log);
    printf("Done\n");

    return 0;
}
