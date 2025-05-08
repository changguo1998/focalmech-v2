cmake_minimum_required(VERSION 3.22)
project(FMv2 LANGUAGES C CUDA)

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
# enable_language(CUDA)
# include(CheckLanguage)
# find_package(CUDAToolkit)
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

add_executable(database_io test/database_io.c src/io.c src/types.c include/io.h include/types.h)
target_include_directories(database_io PRIVATE include)
install(TARGETS database_io)
