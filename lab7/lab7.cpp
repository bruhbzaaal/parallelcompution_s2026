#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

using namespace std;

// ================= SCALAR =================
void prewitt_scalar(const vector<uint8_t>& input, vector<uint8_t>& output, int w, int h) {
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {

            int gx =
                -input[(y - 1) * w + (x - 1)] + input[(y - 1) * w + (x + 1)] +
                -input[y * w + (x - 1)]     + input[y * w + (x + 1)] +
                -input[(y + 1) * w + (x - 1)] + input[(y + 1) * w + (x + 1)];

            int gy =
                 input[(y - 1) * w + (x - 1)] + input[(y - 1) * w + x] + input[(y - 1) * w + (x + 1)] -
                 input[(y + 1) * w + (x - 1)] - input[(y + 1) * w + x] - input[(y + 1) * w + (x + 1)];

            int val = (int)sqrt(gx * gx + gy * gy);
            if (val > 255) val = 255;

            output[y * w + x] = (uint8_t)val;
        }
    }
}

// ================= SIMD (SSE) =================
void prewitt_simd(const vector<uint8_t>& input, vector<uint8_t>& output, int w, int h) {

    for (int y = 1; y < h - 1; y++) {

        for (int x = 1; x < w - 17; x += 16) {

            __m128i top_l = _mm_loadu_si128((__m128i*)&input[(y - 1) * w + x - 1]);
            __m128i top_r = _mm_loadu_si128((__m128i*)&input[(y - 1) * w + x + 1]);

            __m128i mid_l = _mm_loadu_si128((__m128i*)&input[y * w + x - 1]);
            __m128i mid_r = _mm_loadu_si128((__m128i*)&input[y * w + x + 1]);

            __m128i bot_l = _mm_loadu_si128((__m128i*)&input[(y + 1) * w + x - 1]);
            __m128i bot_r = _mm_loadu_si128((__m128i*)&input[(y + 1) * w + x + 1]);

            // gx = right - left
            __m128i gx = _mm_sub_epi8(
                _mm_add_epi8(_mm_add_epi8(top_r, mid_r), bot_r),
                _mm_add_epi8(_mm_add_epi8(top_l, mid_l), bot_l)
            );

            __m128i top = _mm_loadu_si128((__m128i*)&input[(y - 1) * w + x]);
            __m128i bot = _mm_loadu_si128((__m128i*)&input[(y + 1) * w + x]);

            __m128i gy = _mm_sub_epi8(top, bot);

            // расширение до int16
            __m128i gx_lo = _mm_unpacklo_epi8(gx, _mm_setzero_si128());
            __m128i gx_hi = _mm_unpackhi_epi8(gx, _mm_setzero_si128());

            __m128i gy_lo = _mm_unpacklo_epi8(gy, _mm_setzero_si128());
            __m128i gy_hi = _mm_unpackhi_epi8(gy, _mm_setzero_si128());

            // квадрат
            __m128i gx2_lo = _mm_mullo_epi16(gx_lo, gx_lo);
            __m128i gx2_hi = _mm_mullo_epi16(gx_hi, gx_hi);

            __m128i gy2_lo = _mm_mullo_epi16(gy_lo, gy_lo);
            __m128i gy2_hi = _mm_mullo_epi16(gy_hi, gy_hi);

            __m128i sum_lo = _mm_add_epi16(gx2_lo, gy2_lo);
            __m128i sum_hi = _mm_add_epi16(gx2_hi, gy2_hi);

            // приближённый sqrt (без float для простоты)
            __m128i result = _mm_packus_epi16(sum_lo, sum_hi);

            _mm_storeu_si128((__m128i*)&output[y * w + x], result);
        }
    }
}

// ================= MAIN =================
int main() {
    const int w = 1024;
    const int h = 1024;

    vector<uint8_t> input(w * h);
    vector<uint8_t> out1(w * h);
    vector<uint8_t> out2(w * h);

    // случайное изображение
    for (int i = 0; i < w * h; i++)
        input[i] = rand() % 256;

    auto t1 = chrono::high_resolution_clock::now();
    prewitt_scalar(input, out1, w, h);
    auto t2 = chrono::high_resolution_clock::now();

    prewitt_simd(input, out2, w, h);
    auto t3 = chrono::high_resolution_clock::now();

    double scalar_time = chrono::duration<double, milli>(t2 - t1).count();
    double simd_time = chrono::duration<double, milli>(t3 - t2).count();

    cout << "Scalar: " << scalar_time << " ms\n";
    cout << "SIMD:   " << simd_time << " ms\n";
    cout << "Speedup: " << scalar_time / simd_time << "x\n";

    return 0;
}
