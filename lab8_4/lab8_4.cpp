#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            cout << "CUDA error: "                            \
                 << cudaGetErrorString(err)                   \
                 << " at line " << __LINE__ << endl;          \
            exit(1);                                          \
        }                                                     \
    } while (0)

const int TILE = 16;

void fillMatrix(float* A, int N)
{
    for (int i = 0; i < N * N; i++)
        A[i] = static_cast<float>((rand() % 100) / 10.0);
}

void multiplyCPU(float* A, float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];

            C[i * N + j] = sum;
        }
    }
}

bool compareMatrices(float* A, float* B, int N)
{
    for (int i = 0; i < N * N; i++)
    {
        if (fabs(A[i] - B[i]) > 0.01f)
            return false;
    }

    return true;
}

__global__ void matrixMulNaive(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];

        C[row * N + col] = sum;
    }
}

__global__ void matrixMulRowCached(float* A, float* B, float* C, int N)
{
    extern __shared__ float rowA[];

    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < N && col < N)
    {
        rowA[col] = A[row * N + col];
    }

    __syncthreads();

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
            sum += rowA[k] * B[k * N + col];

        C[row * N + col] = sum;
    }
}

__global__ void matrixMulColumnCached(float* A, float* B, float* C, int N)
{
    extern __shared__ float colB[];

    int col = blockIdx.x;
    int row = threadIdx.x;

    if (row < N && col < N)
    {
        colB[row] = B[row * N + col];
    }

    __syncthreads();

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * colB[k];

        C[row * N + col] = sum;
    }
}

__global__ void matrixMulTiled(float* A, float* B, float* C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++)
    {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    C[row * N + col] = sum;
}

float runNaive(float* d_A, float* d_B, float* d_C, int N)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    matrixMulNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float runRowCached(float* d_A, float* d_B, float* d_C, int N)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    matrixMulRowCached<<<N, N, N * sizeof(float)>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float runColumnCached(float* d_A, float* d_B, float* d_C, int N)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    matrixMulColumnCached<<<N, N, N * sizeof(float)>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float runTiled(float* d_A, float* d_B, float* d_C, int N)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    matrixMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

double gflops(int N, float ms)
{
    double operations = 2.0 * N * N * N;
    double seconds = ms / 1000.0;
    return operations / seconds / 1e9;
}

int main()
{
    srand(1);

    int sizes[] = {128, 256, 512};

    cout << fixed << setprecision(4);

    cout << "CUDA matrix multiplication benchmark" << endl;
    cout << "Tile size: " << TILE << " x " << TILE << endl << endl;

    cout << setw(8) << "N"
         << setw(18) << "Algorithm"
         << setw(15) << "Time, ms"
         << setw(15) << "GFLOPS"
         << setw(12) << "Correct"
         << endl;

    cout << "--------------------------------------------------------------------" << endl;

    for (int s = 0; s < 3; s++)
    {
        int N = sizes[s];
        size_t bytes = N * N * sizeof(float);

        float* h_A = new float[N * N];
        float* h_B = new float[N * N];
        float* h_C_cpu = new float[N * N];
        float* h_C_gpu = new float[N * N];

        fillMatrix(h_A, N);
        fillMatrix(h_B, N);

        multiplyCPU(h_A, h_B, h_C_cpu, N);

        float* d_A;
        float* d_B;
        float* d_C;

        CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        float timeNaive = runNaive(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        bool okNaive = compareMatrices(h_C_cpu, h_C_gpu, N);

        cout << setw(8) << N
             << setw(18) << "Naive"
             << setw(15) << timeNaive
             << setw(15) << gflops(N, timeNaive)
             << setw(12) << (okNaive ? "OK" : "ERROR")
             << endl;

        float timeRow = runRowCached(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        bool okRow = compareMatrices(h_C_cpu, h_C_gpu, N);

        cout << setw(8) << N
             << setw(18) << "Row cached"
             << setw(15) << timeRow
             << setw(15) << gflops(N, timeRow)
             << setw(12) << (okRow ? "OK" : "ERROR")
             << endl;

        float timeCol = runColumnCached(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        bool okCol = compareMatrices(h_C_cpu, h_C_gpu, N);

        cout << setw(8) << N
             << setw(18) << "Column cached"
             << setw(15) << timeCol
             << setw(15) << gflops(N, timeCol)
             << setw(12) << (okCol ? "OK" : "ERROR")
             << endl;

        float timeTiled = runTiled(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        bool okTiled = compareMatrices(h_C_cpu, h_C_gpu, N);

        cout << setw(8) << N
             << setw(18) << "Tiled shared"
             << setw(15) << timeTiled
             << setw(15) << gflops(N, timeTiled)
             << setw(12) << (okTiled ? "OK" : "ERROR")
             << endl;

        cout << "--------------------------------------------------------------------" << endl;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        delete[] h_A;
        delete[] h_B;
        delete[] h_C_cpu;
        delete[] h_C_gpu;
    }

    cout << endl;
    cout << "Benchmark completed successfully." << endl;

    return 0;
}
