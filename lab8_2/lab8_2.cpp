#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    cout << "CUDA device count: " << deviceCount << endl << endl;

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, i);

        cout << "Device " << i << ":" << endl;
        cout << "Name: " << dp.name << endl;

        cout << "Total global memory: " 
             << dp.totalGlobalMem / (1024 * 1024) << " MB" << endl;

        cout << "Shared memory per block: " 
             << dp.sharedMemPerBlock / 1024 << " KB" << endl;

        cout << "Constant memory: " 
             << dp.totalConstMem / 1024 << " KB" << endl;

        cout << "Registers per block: " 
             << dp.regsPerBlock << endl;

        cout << "Warp size: " 
             << dp.warpSize << endl;

        cout << "Max threads per block: " 
             << dp.maxThreadsPerBlock << endl;

        cout << "Multiprocessor count: " 
             << dp.multiProcessorCount << endl;

        cout << "Clock rate: " 
             << dp.clockRate / 1000 << " MHz" << endl;

        cout << "Memory clock rate: " 
             << dp.memoryClockRate / 1000 << " MHz" << endl;

        cout << "Memory bus width: " 
             << dp.memoryBusWidth << " bits" << endl;

        cout << "L2 cache size: " 
             << dp.l2CacheSize / 1024 << " KB" << endl;

        cout << "Compute capability: " 
             << dp.major << "." << dp.minor << endl;

        cout << endl;
    }

    return 0;
}
