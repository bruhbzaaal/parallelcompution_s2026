#include <windows.h>
#include <intrin.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;

struct Stats {
    double minVal;
    double avgVal;
    double variance;
    double sigma;
    double delta;
    double ciLeft;
    double ciRight;
    vector<double> filtered;
};

bool isPrime(int x) {
    if (x < 2) return false;
    if (x == 2) return true;
    if (x % 2 == 0) return false;

    int limit = static_cast<int>(sqrt(x));
    for (int d = 3; d <= limit; d += 2) {
        if (x % d == 0) return false;
    }

    return true;
}

int countPrimes(int n) {
    int cnt = 0;

    for (int i = 2; i <= n; ++i) {
        if (isPrime(i)) cnt++;
    }

    return cnt;
}

double mean(const vector<double>& v) {
    double s = accumulate(v.begin(), v.end(), 0.0);
    return s / v.size();
}

double variance(const vector<double>& v, double avg) {
    double s = 0;

    for (double x : v) {
        double d = x - avg;
        s += d * d;
    }

    return s / v.size();
}

Stats calculateStats(const vector<double>& data) {

    Stats st{};

    st.minVal = *min_element(data.begin(), data.end());

    double avg = mean(data);
    double var = variance(data, avg);
    double sig = sqrt(var);

    vector<double> filtered;

    for (double x : data) {
        if (fabs(x - avg) <= 3 * sig)
            filtered.push_back(x);
    }

    if (filtered.empty())
        filtered = data;

    double avg2 = mean(filtered);
    double var2 = variance(filtered, avg2);
    double sig2 = sqrt(var2);

    double delta = 1.96 * sig2 / sqrt(filtered.size());

    st.avgVal = avg2;
    st.variance = var2;
    st.sigma = sig2;
    st.delta = delta;

    st.ciLeft = avg2 - delta;
    st.ciRight = avg2 + delta;

    st.filtered = filtered;

    return st;
}

void saveCSV(const string& filename,
    const vector<double>& tickData,
    const vector<double>& qpcData,
    const vector<double>& tscData) {

    ofstream fout(filename);

    fout << "Measure,GetTickCount_ms,QueryPerformanceCounter_ms,RDTSC_ms\n";

    for (size_t i = 0; i < tickData.size(); i++) {

        fout << i + 1 << ","
            << tickData[i] << ","
            << qpcData[i] << ","
            << tscData[i] << "\n";
    }

    fout.close();
}

void printStats(const string& title, const Stats& st, const string& unit) {

    cout << "\n=== " << title << " ===\n";

    cout << "Minimum time: "
        << st.minVal << " " << unit << endl;

    cout << "Average time (after outlier removal): "
        << st.avgVal << " " << unit << endl;

    cout << "Variance: "
        << st.variance << endl;

    cout << "Standard deviation: "
        << st.sigma << " " << unit << endl;

    cout << "95% confidence interval: ["
        << st.ciLeft << "; "
        << st.ciRight << "] "
        << unit << endl;

    cout << "Final result: "
        << st.avgVal << " +- "
        << st.delta << " "
        << unit << endl;

    cout << "Number of samples after filtering: "
        << st.filtered.size() << endl;
}

int main() {

    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    const int N = 200000;
    const int K = 30;

    vector<double> tickTimes;
    vector<double> qpcTimes;
    vector<double> tscTimes;

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    cout << fixed << setprecision(6);

    cout << "Algorithm: counting prime numbers up to N\n";
    cout << "N = " << N << endl;
    cout << "Number of measurements K = " << K << endl << endl;

    volatile int result = 0;

    for (int i = 0; i < K; i++) {

        DWORD startTick = GetTickCount();

        result = countPrimes(N);

        DWORD endTick = GetTickCount();

        double tickMs = endTick - startTick;

        tickTimes.push_back(tickMs);


        LARGE_INTEGER startQpc, endQpc;

        QueryPerformanceCounter(&startQpc);

        result = countPrimes(N);

        QueryPerformanceCounter(&endQpc);

        double qpcMs =
            (endQpc.QuadPart - startQpc.QuadPart) * 1000.0 / freq.QuadPart;

        qpcTimes.push_back(qpcMs);


        unsigned __int64 startTsc = __rdtsc();

        result = countPrimes(N);

        unsigned __int64 endTsc = __rdtsc();

        LARGE_INTEGER q1, q2;

        QueryPerformanceCounter(&q1);
        Sleep(1000);
        QueryPerformanceCounter(&q2);

        unsigned __int64 t1 = __rdtsc();
        Sleep(1000);
        unsigned __int64 t2 = __rdtsc();

        double cpuHz =
            (double)(t2 - t1) /
            ((double)(q2.QuadPart - q1.QuadPart) / freq.QuadPart);

        double tscMs =
            ((double)(endTsc - startTsc) / cpuHz) * 1000.0;

        tscTimes.push_back(tscMs);

        cout << "Measurement " << i + 1
            << ": GetTickCount = " << tickMs << " ms, "
            << "QueryPerformanceCounter = " << qpcMs << " ms, "
            << "RDTSC = " << tscMs << " ms"
            << endl;
    }

    cout << "\nComputation result (to prevent optimization): "
        << result << endl;

    Stats tickStats = calculateStats(tickTimes);
    Stats qpcStats = calculateStats(qpcTimes);
    Stats tscStats = calculateStats(tscTimes);

    printStats("GetTickCount", tickStats, "ms");
    printStats("QueryPerformanceCounter", qpcStats, "ms");
    printStats("RDTSC", tscStats, "ms");

    saveCSV("results.csv", tickTimes, qpcTimes, tscTimes);

    cout << "\nFile results.csv saved." << endl;
    cout << "Use it to build tables and histograms." << endl;

    return 0;
}
