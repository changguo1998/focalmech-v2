#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

typedef long long Int64;

typedef double Float64;

__global__ void assign(Int64 *iv, Float64 *fv, Int64 n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;

    iv[idx] *= 2;
    fv[idx] *= 2.0;
    return;
}

void call_knl(Int64 *iv, Float64 *fv, Int64 n)
{
    int b = 32;
    int g = (n - 1) / b + 1;
    assign<<<g, b>>>(iv, fv, n);
    cudaDeviceSynchronize();
}

#define L 1024

int main()
{
    Int64 ivec[L], i, *ivec_d;
    Float64 fvec[L], *fvec_d;

    for (i = 0; i < L; i++)
    {
        ivec[i] = i;
        fvec[i] = i;
    }
    cudaMalloc(&ivec_d, L * sizeof(Int64));
    cudaMemcpy(ivec_d, ivec, L * sizeof(Int64), cudaMemcpyHostToDevice);
    cudaMalloc(&fvec_d, L * sizeof(Float64));
    cudaMemcpy(fvec_d, fvec, L * sizeof(Float64), cudaMemcpyHostToDevice);
    call_knl(ivec_d, fvec_d, L);
    cudaMemcpy(ivec, ivec_d, L * sizeof(Int64), cudaMemcpyDeviceToHost);
    cudaMemcpy(fvec, fvec_d, L * sizeof(Float64), cudaMemcpyDeviceToHost);

    for(i=0; i<5; i++){
        printf("%d -- %d -- %f\n", i, ivec[i], fvec[i]);
    }
    cudaFree(ivec_d);
    cudaFree(fvec_d);
    return 0;
}
