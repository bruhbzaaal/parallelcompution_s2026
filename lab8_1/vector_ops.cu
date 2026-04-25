#include <cuda.h>
#include <cuda_runtime.h>

__global__ void VecAddKernel(float* a, float* b, float* c, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

extern "C" void vec_add_cuda(float* a, float* b, float* c, int n)
{
    int sizeInBytes = n * sizeof(float);

    float* a_gpu = nullptr;
    float* b_gpu = nullptr;
    float* c_gpu = nullptr;

    cudaMalloc((void**)&a_gpu, sizeInBytes);
    cudaMalloc((void**)&b_gpu, sizeInBytes);
    cudaMalloc((void**)&c_gpu, sizeInBytes);

    cudaMemcpy(a_gpu, a, sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeInBytes, cudaMemcpyHostToDevice);

    dim3 threads(512, 1);
    dim3 blocks((n + threads.x - 1) / threads.x, 1);

    VecAddKernel<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c, c_gpu, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}
