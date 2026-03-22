#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <functional>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <fstream>
#include <string>

using Matrix = std::vector<float>;

// ============================================================
// Build configuration name
// ============================================================
#ifdef _DEBUG
const char* BUILD_CONFIG = "Debug";
#else
const char* BUILD_CONFIG = "Release";
#endif

// ============================================================
// Utility
// ============================================================

int idx(int row, int col, int N)
{
    return row * N + col;
}

void fillMatrixRandom(Matrix& M, int N, float minValue = 0.0f, float maxValue = 10.0f)
{
    static std::mt19937 gen(
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count())
    );
    std::uniform_real_distribution<float> dist(minValue, maxValue);

    M.resize(N * N);
    for (float& x : M)
        x = dist(gen);
}

bool compareMatrices(const Matrix& A, const Matrix& B, int N, float eps = 1e-3f)
{
    if (A.size() != B.size() || static_cast<int>(A.size()) != N * N)
        return false;

    for (int i = 0; i < N * N; ++i)
    {
        if (std::fabs(A[i] - B[i]) > eps)
            return false;
    }
    return true;
}

void transposeMatrix(const Matrix& B, Matrix& BT, int N)
{
    BT.resize(N * N);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            BT[idx(j, i, N)] = B[idx(i, j, N)];
}

double measureTime(const std::function<void()>& func, int repeats = 1)
{
    double bestTime = std::numeric_limits<double>::max();

    for (int r = 0; r < repeats; ++r)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        if (elapsed < bestTime)
            bestTime = elapsed;
    }

    return bestTime;
}

double calcPerformanceGFLOPS(int N, double timeSec)
{
    if (timeSec <= 0.0)
        return 0.0;

    const double operations = 2.0 * static_cast<double>(N) * N * N;
    return operations / timeSec / 1e9;
}

void printResult(const std::string& name, int N, double timeSec, double gflops, bool ok = true)
{
    std::cout << std::left << std::setw(26) << name
        << " N=" << std::setw(5) << N
        << " time=" << std::setw(12) << std::fixed << std::setprecision(6) << timeSec
        << " GFLOPS=" << std::setw(10) << std::setprecision(3) << gflops
        << " ok=" << (ok ? "OK" : "FAIL")
        << '\n';
}

// ============================================================
// Multiplication algorithms
// ============================================================

void multiplyClassic(const Matrix& A, const Matrix& B, Matrix& C, int N)
{
    C.assign(N * N, 0.0f);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[idx(i, k, N)] * B[idx(k, j, N)];

            C[idx(i, j, N)] = sum;
        }
    }
}

void multiplyTranspose(const Matrix& A, const Matrix& BT, Matrix& C, int N)
{
    C.assign(N * N, 0.0f);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[idx(i, k, N)] * BT[idx(j, k, N)];

            C[idx(i, j, N)] = sum;
        }
    }
}

// Buffered multiplication with unroll factor M.
// Supported M: 1, 2, 4, 8, 16.
void multiplyBufferedUnrolled(const Matrix& A, const Matrix& B, Matrix& C, int N, int M)
{
    C.assign(N * N, 0.0f);
    std::vector<float> tmp(N);

    for (int j = 0; j < N; ++j)
    {
        for (int k = 0; k < N; ++k)
            tmp[k] = B[idx(k, j, N)];

        for (int i = 0; i < N; ++i)
        {
            const float* aRow = &A[idx(i, 0, N)];
            float sum = 0.0f;
            int k = 0;

            switch (M)
            {
            case 16:
                for (; k + 15 < N; k += 16)
                {
                    sum += aRow[k + 0] * tmp[k + 0];
                    sum += aRow[k + 1] * tmp[k + 1];
                    sum += aRow[k + 2] * tmp[k + 2];
                    sum += aRow[k + 3] * tmp[k + 3];
                    sum += aRow[k + 4] * tmp[k + 4];
                    sum += aRow[k + 5] * tmp[k + 5];
                    sum += aRow[k + 6] * tmp[k + 6];
                    sum += aRow[k + 7] * tmp[k + 7];
                    sum += aRow[k + 8] * tmp[k + 8];
                    sum += aRow[k + 9] * tmp[k + 9];
                    sum += aRow[k + 10] * tmp[k + 10];
                    sum += aRow[k + 11] * tmp[k + 11];
                    sum += aRow[k + 12] * tmp[k + 12];
                    sum += aRow[k + 13] * tmp[k + 13];
                    sum += aRow[k + 14] * tmp[k + 14];
                    sum += aRow[k + 15] * tmp[k + 15];
                }
                break;

            case 8:
                for (; k + 7 < N; k += 8)
                {
                    sum += aRow[k + 0] * tmp[k + 0];
                    sum += aRow[k + 1] * tmp[k + 1];
                    sum += aRow[k + 2] * tmp[k + 2];
                    sum += aRow[k + 3] * tmp[k + 3];
                    sum += aRow[k + 4] * tmp[k + 4];
                    sum += aRow[k + 5] * tmp[k + 5];
                    sum += aRow[k + 6] * tmp[k + 6];
                    sum += aRow[k + 7] * tmp[k + 7];
                }
                break;

            case 4:
                for (; k + 3 < N; k += 4)
                {
                    sum += aRow[k + 0] * tmp[k + 0];
                    sum += aRow[k + 1] * tmp[k + 1];
                    sum += aRow[k + 2] * tmp[k + 2];
                    sum += aRow[k + 3] * tmp[k + 3];
                }
                break;

            case 2:
                for (; k + 1 < N; k += 2)
                {
                    sum += aRow[k + 0] * tmp[k + 0];
                    sum += aRow[k + 1] * tmp[k + 1];
                }
                break;

            case 1:
            default:
                break;
            }

            for (; k < N; ++k)
                sum += aRow[k] * tmp[k];

            C[idx(i, j, N)] = sum;
        }
    }
}

// Blocked multiplication with unroll factor M in the inner loop.
// Supported M: 1, 2, 4, 8, 16.
void multiplyBlockedUnrolled(const Matrix& A, const Matrix& B, Matrix& C, int N, int blockSize, int M)
{
    C.assign(N * N, 0.0f);

    for (int ii = 0; ii < N; ii += blockSize)
    {
        for (int jj = 0; jj < N; jj += blockSize)
        {
            for (int kk = 0; kk < N; kk += blockSize)
            {
                const int iMax = std::min(ii + blockSize, N);
                const int jMax = std::min(jj + blockSize, N);
                const int kMax = std::min(kk + blockSize, N);

                for (int i = ii; i < iMax; ++i)
                {
                    for (int j = jj; j < jMax; ++j)
                    {
                        float sum = C[idx(i, j, N)];
                        int k = kk;

                        switch (M)
                        {
                        case 16:
                            for (; k + 15 < kMax; k += 16)
                            {
                                sum += A[idx(i, k + 0, N)] * B[idx(k + 0, j, N)];
                                sum += A[idx(i, k + 1, N)] * B[idx(k + 1, j, N)];
                                sum += A[idx(i, k + 2, N)] * B[idx(k + 2, j, N)];
                                sum += A[idx(i, k + 3, N)] * B[idx(k + 3, j, N)];
                                sum += A[idx(i, k + 4, N)] * B[idx(k + 4, j, N)];
                                sum += A[idx(i, k + 5, N)] * B[idx(k + 5, j, N)];
                                sum += A[idx(i, k + 6, N)] * B[idx(k + 6, j, N)];
                                sum += A[idx(i, k + 7, N)] * B[idx(k + 7, j, N)];
                                sum += A[idx(i, k + 8, N)] * B[idx(k + 8, j, N)];
                                sum += A[idx(i, k + 9, N)] * B[idx(k + 9, j, N)];
                                sum += A[idx(i, k + 10, N)] * B[idx(k + 10, j, N)];
                                sum += A[idx(i, k + 11, N)] * B[idx(k + 11, j, N)];
                                sum += A[idx(i, k + 12, N)] * B[idx(k + 12, j, N)];
                                sum += A[idx(i, k + 13, N)] * B[idx(k + 13, j, N)];
                                sum += A[idx(i, k + 14, N)] * B[idx(k + 14, j, N)];
                                sum += A[idx(i, k + 15, N)] * B[idx(k + 15, j, N)];
                            }
                            break;

                        case 8:
                            for (; k + 7 < kMax; k += 8)
                            {
                                sum += A[idx(i, k + 0, N)] * B[idx(k + 0, j, N)];
                                sum += A[idx(i, k + 1, N)] * B[idx(k + 1, j, N)];
                                sum += A[idx(i, k + 2, N)] * B[idx(k + 2, j, N)];
                                sum += A[idx(i, k + 3, N)] * B[idx(k + 3, j, N)];
                                sum += A[idx(i, k + 4, N)] * B[idx(k + 4, j, N)];
                                sum += A[idx(i, k + 5, N)] * B[idx(k + 5, j, N)];
                                sum += A[idx(i, k + 6, N)] * B[idx(k + 6, j, N)];
                                sum += A[idx(i, k + 7, N)] * B[idx(k + 7, j, N)];
                            }
                            break;

                        case 4:
                            for (; k + 3 < kMax; k += 4)
                            {
                                sum += A[idx(i, k + 0, N)] * B[idx(k + 0, j, N)];
                                sum += A[idx(i, k + 1, N)] * B[idx(k + 1, j, N)];
                                sum += A[idx(i, k + 2, N)] * B[idx(k + 2, j, N)];
                                sum += A[idx(i, k + 3, N)] * B[idx(k + 3, j, N)];
                            }
                            break;

                        case 2:
                            for (; k + 1 < kMax; k += 2)
                            {
                                sum += A[idx(i, k + 0, N)] * B[idx(k + 0, j, N)];
                                sum += A[idx(i, k + 1, N)] * B[idx(k + 1, j, N)];
                            }
                            break;

                        case 1:
                        default:
                            break;
                        }

                        for (; k < kMax; ++k)
                            sum += A[idx(i, k, N)] * B[idx(k, j, N)];

                        C[idx(i, j, N)] = sum;
                    }
                }
            }
        }
    }
}

// ============================================================
// Experiment helpers
// ============================================================

struct PerfResult
{
    double timeSec = 0.0;
    double gflops = 0.0;
    bool ok = true;
};

PerfResult runClassic(const Matrix& A, const Matrix& B, Matrix& C, int N, int repeats)
{
    PerfResult r;
    r.timeSec = measureTime([&]() { multiplyClassic(A, B, C, N); }, repeats);
    r.gflops = calcPerformanceGFLOPS(N, r.timeSec);
    r.ok = true;
    return r;
}

PerfResult runTransposeOnly(const Matrix& A, const Matrix& BT, const Matrix& reference, Matrix& C, int N, int repeats)
{
    PerfResult r;
    r.timeSec = measureTime([&]() { multiplyTranspose(A, BT, C, N); }, repeats);
    r.gflops = calcPerformanceGFLOPS(N, r.timeSec);
    r.ok = compareMatrices(reference, C, N);
    return r;
}

PerfResult runTransposeWithPrep(const Matrix& A, const Matrix& B, const Matrix& reference, Matrix& C, int N, int repeats)
{
    PerfResult r;
    r.timeSec = measureTime([&]()
        {
            Matrix BTlocal;
            transposeMatrix(B, BTlocal, N);
            multiplyTranspose(A, BTlocal, C, N);
        }, repeats);
    r.gflops = calcPerformanceGFLOPS(N, r.timeSec);
    r.ok = compareMatrices(reference, C, N);
    return r;
}

PerfResult runBuffered(const Matrix& A, const Matrix& B, const Matrix& reference, Matrix& C, int N, int repeats, int M)
{
    PerfResult r;
    r.timeSec = measureTime([&]() { multiplyBufferedUnrolled(A, B, C, N, M); }, repeats);
    r.gflops = calcPerformanceGFLOPS(N, r.timeSec);
    r.ok = compareMatrices(reference, C, N);
    return r;
}

PerfResult runBlocked(const Matrix& A, const Matrix& B, const Matrix& reference, Matrix& C, int N, int repeats, int S, int M)
{
    PerfResult r;
    r.timeSec = measureTime([&]() { multiplyBlockedUnrolled(A, B, C, N, S, M); }, repeats);
    r.gflops = calcPerformanceGFLOPS(N, r.timeSec);
    r.ok = compareMatrices(reference, C, N);
    return r;
}

void writeCsvHeader(std::ofstream& file)
{
    file << "BuildConfig,Section,N,Algorithm,M,S,TimeSec,GFLOPS,Correct\n";
}

// ============================================================
// Main
// ============================================================

int main()
{
    // Sizes for points 5, 6, and 9
    const std::vector<int> sizes = { 64, 128, 256, 512, 1024 };

    // Unroll factors required by the lab
    const std::vector<int> unrollFactors = { 1, 2, 4, 8, 16 };

    // Block sizes required for blocked algorithm study
    const std::vector<int> blockSizes = { 4, 8, 16, 32, 64 };

    // Chosen N for points 7 and 8
    const int selectedNBuffered = 512;
    const int selectedNBlocked = 512;

#ifdef _DEBUG
    const int repeats = 1;
#else
    const int repeats = 3;
#endif

    std::ofstream csv("labb_results.csv");
    if (!csv.is_open())
    {
        std::cerr << "Cannot open labb_results.csv\n";
        return 1;
    }
    writeCsvHeader(csv);

    std::cout << "Build=" << BUILD_CONFIG << '\n';
    std::cout << "Start experiments\n\n";

    // --------------------------------------------------------
    // Point 5 + Point 6 baseline across N
    // --------------------------------------------------------
    std::cout << "Section 5 and 6: classic and transpose variants\n";

    for (int N : sizes)
    {
        Matrix A, B, BT;
        Matrix Cclassic, CtransposeOnly, CtransposePrep;

        fillMatrixRandom(A, N);
        fillMatrixRandom(B, N);
        transposeMatrix(B, BT, N);

        PerfResult classicRes = runClassic(A, B, Cclassic, N, repeats);
        printResult("Classic", N, classicRes.timeSec, classicRes.gflops, true);
        csv << BUILD_CONFIG << ",Main,N" << N << ",Classic,1,0," << classicRes.timeSec << "," << classicRes.gflops << ",OK\n";

        PerfResult transposeOnlyRes = runTransposeOnly(A, BT, Cclassic, CtransposeOnly, N, repeats);
        printResult("TransposeOnly", N, transposeOnlyRes.timeSec, transposeOnlyRes.gflops, transposeOnlyRes.ok);
        csv << BUILD_CONFIG << ",Main,N" << N << ",TransposeOnly,1,0," << transposeOnlyRes.timeSec << "," << transposeOnlyRes.gflops << "," << (transposeOnlyRes.ok ? "OK" : "FAIL") << "\n";

        PerfResult transposePrepRes = runTransposeWithPrep(A, B, Cclassic, CtransposePrep, N, repeats);
        printResult("TransposeWithPrep", N, transposePrepRes.timeSec, transposePrepRes.gflops, transposePrepRes.ok);
        csv << BUILD_CONFIG << ",Main,N" << N << ",TransposeWithPrep,1,0," << transposePrepRes.timeSec << "," << transposePrepRes.gflops << "," << (transposePrepRes.ok ? "OK" : "FAIL") << "\n";

        std::cout << '\n';
    }

    // --------------------------------------------------------
    // Point 7: buffered algorithm, Preal(M) for chosen N
    // --------------------------------------------------------
    std::cout << "Section 7: buffered algorithm tuning by M\n";

    int bestBufferedM = 1;
    double bestBufferedPerf = -1.0;

    {
        const int N = selectedNBuffered;
        Matrix A, B, Cclassic, Cbuffered;

        fillMatrixRandom(A, N);
        fillMatrixRandom(B, N);

        PerfResult classicRes = runClassic(A, B, Cclassic, N, repeats);
        printResult("BufferedBaselineClassic", N, classicRes.timeSec, classicRes.gflops, true);
        csv << BUILD_CONFIG << ",BufferedTune," << N << ",Classic,1,0," << classicRes.timeSec << "," << classicRes.gflops << ",OK\n";

        for (int M : unrollFactors)
        {
            PerfResult r = runBuffered(A, B, Cclassic, Cbuffered, N, repeats, M);
            printResult("Buffered", N, r.timeSec, r.gflops, r.ok);
            std::cout << "  M=" << M << '\n';
            csv << BUILD_CONFIG << ",BufferedTune," << N << ",Buffered," << M << ",0," << r.timeSec << "," << r.gflops << "," << (r.ok ? "OK" : "FAIL") << "\n";

            if (r.ok && r.gflops > bestBufferedPerf)
            {
                bestBufferedPerf = r.gflops;
                bestBufferedM = M;
            }
        }

        std::cout << "BestBufferedM=" << bestBufferedM
            << " BestBufferedGFLOPS=" << std::fixed << std::setprecision(3) << bestBufferedPerf
            << "\n\n";
    }

    // --------------------------------------------------------
    // Point 8: blocked algorithm, Preal(M,S) for chosen N
    // --------------------------------------------------------
    std::cout << "Section 8: blocked algorithm tuning by M and S\n";

    int bestBlockedM = 1;
    int bestBlockedS = blockSizes.front();
    double bestBlockedPerf = -1.0;

    {
        const int N = selectedNBlocked;
        Matrix A, B, Cclassic, Cblocked;

        fillMatrixRandom(A, N);
        fillMatrixRandom(B, N);

        PerfResult classicRes = runClassic(A, B, Cclassic, N, repeats);
        printResult("BlockedBaselineClassic", N, classicRes.timeSec, classicRes.gflops, true);
        csv << BUILD_CONFIG << ",BlockedTune," << N << ",Classic,1,0," << classicRes.timeSec << "," << classicRes.gflops << ",OK\n";

        for (int S : blockSizes)
        {
            if (S > N)
                continue;

            for (int M : unrollFactors)
            {
                PerfResult r = runBlocked(A, B, Cclassic, Cblocked, N, repeats, S, M);
                printResult("Blocked", N, r.timeSec, r.gflops, r.ok);
                std::cout << "  S=" << S << " M=" << M << '\n';
                csv << BUILD_CONFIG << ",BlockedTune," << N << ",Blocked," << M << "," << S << "," << r.timeSec << "," << r.gflops << "," << (r.ok ? "OK" : "FAIL") << "\n";

                if (r.ok && r.gflops > bestBlockedPerf)
                {
                    bestBlockedPerf = r.gflops;
                    bestBlockedM = M;
                    bestBlockedS = S;
                }
            }
        }

        std::cout << "BestBlockedM=" << bestBlockedM
            << " BestBlockedS=" << bestBlockedS
            << " BestBlockedGFLOPS=" << std::fixed << std::setprecision(3) << bestBlockedPerf
            << "\n\n";
    }

    // --------------------------------------------------------
    // Point 9: P(N) for the 4 best implementations
    //   1) Classic
    //   2) TransposeOnly
    //   3) Buffered(best M)
    //   4) Blocked(best M, best S)
    // --------------------------------------------------------
    std::cout << "Section 9: best implementations across N\n";

    for (int N : sizes)
    {
        Matrix A, B, BT;
        Matrix Cclassic, Ctranspose, Cbuffered, Cblocked;

        fillMatrixRandom(A, N);
        fillMatrixRandom(B, N);
        transposeMatrix(B, BT, N);

        PerfResult classicRes = runClassic(A, B, Cclassic, N, repeats);
        printResult("BestClassic", N, classicRes.timeSec, classicRes.gflops, true);
        csv << BUILD_CONFIG << ",BestAcrossN," << N << ",ClassicBest,1,0," << classicRes.timeSec << "," << classicRes.gflops << ",OK\n";

        PerfResult transposeRes = runTransposeOnly(A, BT, Cclassic, Ctranspose, N, repeats);
        printResult("BestTranspose", N, transposeRes.timeSec, transposeRes.gflops, transposeRes.ok);
        csv << BUILD_CONFIG << ",BestAcrossN," << N << ",TransposeBest,1,0," << transposeRes.timeSec << "," << transposeRes.gflops << "," << (transposeRes.ok ? "OK" : "FAIL") << "\n";

        PerfResult bufferedRes = runBuffered(A, B, Cclassic, Cbuffered, N, repeats, bestBufferedM);
        printResult("BestBuffered", N, bufferedRes.timeSec, bufferedRes.gflops, bufferedRes.ok);
        csv << BUILD_CONFIG << ",BestAcrossN," << N << ",BufferedBest," << bestBufferedM << ",0," << bufferedRes.timeSec << "," << bufferedRes.gflops << "," << (bufferedRes.ok ? "OK" : "FAIL") << "\n";

        PerfResult blockedRes = runBlocked(A, B, Cclassic, Cblocked, N, repeats, bestBlockedS, bestBlockedM);
        printResult("BestBlocked", N, blockedRes.timeSec, blockedRes.gflops, blockedRes.ok);
        csv << BUILD_CONFIG << ",BestAcrossN," << N << ",BlockedBest," << bestBlockedM << "," << bestBlockedS << "," << blockedRes.timeSec << "," << blockedRes.gflops << "," << (blockedRes.ok ? "OK" : "FAIL") << "\n";

        std::cout << '\n';
    }

    csv.close();

    std::cout << "Done\n";
    std::cout << "BestBufferedM=" << bestBufferedM << '\n';
    std::cout << "BestBlockedM=" << bestBlockedM << '\n';
    std::cout << "BestBlockedS=" << bestBlockedS << '\n';
    std::cout << "CSV=labb_results.csv\n";
    std::cout << "Run this program in Debug and Release separately\n";

    return 0;
}
