#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace std;
using namespace chrono;

const int N = 1'000'000;

// ------------------------
// 1. Обычный C++
// ------------------------
void scalar_add(const float* a, const float* b, float* c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

// ------------------------
// 2. SIMD AVX2
// ------------------------
void simd_add(const float* a, const float* b, float* c) {
    for (int i = 0; i < N; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
}

// ------------------------
// 3. SIMD + развертка x4
// ------------------------
void simd_add_unroll4(const float* a, const float* b, float* c) {
    for (int i = 0; i < N; i += 32) {
        __m256 a1 = _mm256_loadu_ps(a + i);
        __m256 b1 = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(a1, b1));

        __m256 a2 = _mm256_loadu_ps(a + i + 8);
        __m256 b2 = _mm256_loadu_ps(b + i + 8);
        _mm256_storeu_ps(c + i + 8, _mm256_add_ps(a2, b2));

        __m256 a3 = _mm256_loadu_ps(a + i + 16);
        __m256 b3 = _mm256_loadu_ps(b + i + 16);
        _mm256_storeu_ps(c + i + 16, _mm256_add_ps(a3, b3));

        __m256 a4 = _mm256_loadu_ps(a + i + 24);
        __m256 b4 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(c + i + 24, _mm256_add_ps(a4, b4));
    }
}

// ------------------------
// Проверка корректности
// ------------------------
bool check(const vector<float>& a, const vector<float>& b) {
    for (int i = 0; i < N; i++) {
        if (abs(a[i] - b[i]) > 1e-5) return false;
    }
    return true;
}

// ------------------------
// Замер времени
// ------------------------
template<typename Func>
double measure(Func f, const float* a, const float* b, float* c) {
    auto start = high_resolution_clock::now();
    f(a, b, c);
    auto end = high_resolution_clock::now();
    return duration<double, milli>(end - start).count();
}

int main() {
    vector<float> a(N), b(N), c1(N), c2(N), c3(N);

    mt19937 rng(42);
    uniform_real_distribution<float> dist(0, 100);

    for (int i = 0; i < N; i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    double t1 = measure(scalar_add, a.data(), b.data(), c1.data());
    double t2 = measure(simd_add, a.data(), b.data(), c2.data());
    double t3 = measure(simd_add_unroll4, a.data(), b.data(), c3.data());

    cout << "Scalar: " << t1 << " ms\n";
    cout << "SIMD: " << t2 << " ms\n";
    cout << "SIMD unroll x4: " << t3 << " ms\n";

    cout << "\nCheck SIMD: " << (check(c1, c2) ? "OK" : "FAIL");
    cout << "\nCheck SIMD unroll: " << (check(c1, c3) ? "OK" : "FAIL") << endl;

    return 0;
}
