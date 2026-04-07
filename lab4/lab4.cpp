#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

using Clock = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

volatile std::uint64_t g_sink = 0;

// ====================== ГЕНЕРАЦИЯ ДАННЫХ ======================

int* make_data(std::size_t count) {
    int* data = new int[count];
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = i % 1024;
    }
    return data;
}

std::size_t* make_index_array(std::size_t count, std::mt19937_64& rng) {
    std::size_t* idx = new std::size_t[count];

    std::uniform_int_distribution<std::size_t> dist(0, count - 1);
    for (std::size_t i = 0; i < count; ++i) {
        idx[i] = dist(rng);
    }
    return idx;
}

// ====================== АЛГОРИТМЫ ======================

std::uint64_t sequential_sum(int* data, std::size_t n) {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

std::uint64_t random_sum(int* data, std::size_t n, std::mt19937_64& rng) {
    std::uint64_t sum = 0;
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    for (std::size_t i = 0; i < n; ++i) {
        sum += data[dist(rng)];
    }
    return sum;
}

std::uint64_t random_index_sum(int* data, std::size_t* idx, std::size_t n) {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += data[idx[i]];
    }
    return sum;
}

// ====================== ИЗМЕРЕНИЕ ======================

template <typename Func>
double measure(Func func, int repeats) {
    double total = 0;

    // прогрев
    g_sink ^= func();

    for (int i = 0; i < repeats; ++i) {
        auto t1 = Clock::now();
        g_sink ^= func();
        auto t2 = Clock::now();

        auto dt = std::chrono::duration_cast<ns>(t2 - t1).count();
        total += dt;
    }

    return total / repeats;
}

// ====================== ОСНОВНАЯ ПРОГРАММА ======================

int main() {
    const int repeats = 7;
    std::mt19937_64 rng(42);

    std::ofstream csv("results.csv");
    csv << "mode,size_bytes,size_kb,N,avg_total_ns,ns_per_iter\n";
    csv << std::fixed << std::setprecision(4);

    std::cout << "Start...\n";

    // ===== 1 диапазон: до 2 МБ =====
    for (std::size_t kb = 1; kb <= 2048; kb += 1) {
        std::size_t bytes = kb * 1024;
        std::size_t n = bytes / sizeof(int);

        int* data = make_data(n);
        std::size_t* idx = make_index_array(n, rng);

        double t1 = measure([&]() { return sequential_sum(data, n); }, repeats);
        double t2 = measure([&]() { return random_sum(data, n, rng); }, repeats);
        double t3 = measure([&]() { return random_index_sum(data, idx, n); }, repeats);

        csv << "sequential," << bytes << "," << kb << "," << n << "," << t1 << "," << t1 / n << "\n";
        csv << "random," << bytes << "," << kb << "," << n << "," << t2 << "," << t2 / n << "\n";
        csv << "random_index," << bytes << "," << kb << "," << n << "," << t3 << "," << t3 / n << "\n";

        delete[] data;
        delete[] idx;

        std::cout << kb << " KB done\n";
    }

    // ===== 2 диапазон: до 32 МБ =====
    for (std::size_t kb = 2560; kb <= 32768; kb += 512) {
        std::size_t bytes = kb * 1024;
        std::size_t n = bytes / sizeof(int);

        int* data = make_data(n);
        std::size_t* idx = make_index_array(n, rng);

        double t1 = measure([&]() { return sequential_sum(data, n); }, repeats);
        double t2 = measure([&]() { return random_sum(data, n, rng); }, repeats);
        double t3 = measure([&]() { return random_index_sum(data, idx, n); }, repeats);

        csv << "sequential," << bytes << "," << kb << "," << n << "," << t1 << "," << t1 / n << "\n";
        csv << "random," << bytes << "," << kb << "," << n << "," << t2 << "," << t2 / n << "\n";
        csv << "random_index," << bytes << "," << kb << "," << n << "," << t3 << "," << t3 / n << "\n";

        delete[] data;
        delete[] idx;

        std::cout << kb << " KB done\n";
    }

    // ===== 3 диапазон: до 150 МБ =====
    for (std::size_t mb = 35; mb <= 150; mb += 5) {
        std::size_t bytes = mb * 1024 * 1024;
        std::size_t n = bytes / sizeof(int);

        int* data = make_data(n);
        std::size_t* idx = make_index_array(n, rng);

        double t1 = measure([&]() { return sequential_sum(data, n); }, repeats);
        double t2 = measure([&]() { return random_sum(data, n, rng); }, repeats);
        double t3 = measure([&]() { return random_index_sum(data, idx, n); }, repeats);

        csv << "sequential," << bytes << "," << (bytes / 1024) << "," << n << "," << t1 << "," << t1 / n << "\n";
        csv << "random," << bytes << "," << (bytes / 1024) << "," << n << "," << t2 << "," << t2 / n << "\n";
        csv << "random_index," << bytes << "," << (bytes / 1024) << "," << n << "," << t3 << "," << t3 / n << "\n";

        delete[] data;
        delete[] idx;

        std::cout << mb << " MB done\n";
    }

    csv.close();

    std::cout << "Done! results.csv created\n";
    std::cout << "Ignore: " << g_sink << "\n";

    return 0;
}
