cmake_minimum_required(VERSION 3.22)
project(FMv2 LANGUAGES C CUDA)
# check_language(CUDA)
enable_language(CUDA)
include(CheckLanguage)
find_package(CUDAToolkit)

include_directories(include)
file(GLOB myincludes include/*.h)

# add_executable(fm src/fm.c src/kernel.c src/types.c src/io.c ${myincludes})
# target_include_directories(fm PRIVATE include)
# # target_link_libraries(fm PRIVATE m)
# install(TARGETS fm)

# find_package(OpenMP)
# add_executable(fm_omp src/fm_omp.c src/kernel.c src/types.c src/io.c ${myincludes})
# target_include_directories(fm_omp PRIVATE include)
# target_link_libraries(fm_omp PRIVATE OpenMP::OpenMP_C)
# install(TARGETS fm_omp)

# message("USE GPU")
# # check_language(CUDA)
# # !!! change arch of GPU before compiling. The supported arch can be get using `nvcc -h`
# set(CMAKE_CUDA_FLAGS "-rdc=true -arch=sm_90 --expt-relaxed-constexpr")
# add_executable(fm_cuda src/fm_cuda.cu src/kernel.cu src/types.cu src/io.cu ${myincludes})
# set_target_properties(fm_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_compile_definitions(fm_cuda PRIVATE GPU)
# target_include_directories(fm_cuda PRIVATE include ${CUDA_LIBRARIES})
# target_link_libraries(fm_cuda PRIVATE CUDA::cudart)
# install(TARGETS fm_cuda)

# test

set(CMAKE_CUDA_FLAGS "-rdc=true -arch=sm_90 --expt-relaxed-constexpr")
add_executable(database_io "test/database_io.cu" "src/GlobalSetting.cu" "src/GreenFunction.cu" "src/io.cu" "src/Phase.cu" "src/Record.cu" "include/GlobalSetting.h" "include/GreenFunction.h" "include/io.h" "include/Phase.h" "include/Record.h" "include/types.h")
set_target_properties(database_io PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(database_io PRIVATE include ${CUDA_LIBRARIES})
target_link_libraries(database_io PRIVATE CUDA::cudart)
install(TARGETS database_io)
