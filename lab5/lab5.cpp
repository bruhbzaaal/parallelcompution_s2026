#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Nanoseconds = std::chrono::nanoseconds;

static volatile long long g_sink = 0;

struct ResultRow {
    std::string mode;
    std::size_t sizeBytes;
    double sizeKB;
    double sizeMB;
    std::size_t elements;
    long long totalNs;
    double nsPerIter;
};

std::vector<std::size_t> buildTestSizes() {
    std::vector<std::size_t> sizes;

    const std::size_t KB = 1024;
    const std::size_t MB = 1024 * 1024;

    for (std::size_t bytes = 1 * KB; bytes <= 2 * MB; bytes += 1 * KB) {
        sizes.push_back(bytes);
    }

    for (std::size_t bytes = 2 * MB + 512 * KB; bytes <= 32 * MB; bytes += 512 * KB) {
        sizes.push_back(bytes);
    }

    for (std::size_t bytes = 32 * MB + 5 * MB; bytes <= 150 * MB; bytes += 5 * MB) {
        sizes.push_back(bytes);
    }

    return sizes;
}

std::vector<int> createDataArray(std::size_t elements) {
    std::vector<int> data(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        data[i] = static_cast<int>(i % 100);
    }
    return data;
}

std::vector<std::size_t> createRandomIndices(std::size_t elements, std::mt19937_64& rng) {
    std::vector<std::size_t> indices(elements);
    std::uniform_int_distribution<std::size_t> dist(0, elements - 1);

    for (std::size_t i = 0; i < elements; ++i) {
        indices[i] = dist(rng);
    }

    return indices;
}

long long measureSequential(const std::vector<int>& data) {
    volatile long long sum = 0;

    const auto start = Clock::now();
    for (std::size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    const auto end = Clock::now();

    g_sink = sum;
    return std::chrono::duration_cast<Nanoseconds>(end - start).count();
}

long long measureRandom(const std::vector<int>& data, std::mt19937_64& rng) {
    volatile long long sum = 0;
    std::uniform_int_distribution<std::size_t> dist(0, data.size() - 1);

    const auto start = Clock::now();
    for (std::size_t i = 0; i < data.size(); ++i) {
        const std::size_t index = dist(rng);
        sum += data[index];
    }
    const auto end = Clock::now();

    g_sink = sum;
    return std::chrono::duration_cast<Nanoseconds>(end - start).count();
}

long long measureRandomWithIndexArray(
    const std::vector<int>& data,
    const std::vector<std::size_t>& indices
) {
    volatile long long sum = 0;

    const auto start = Clock::now();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        sum += data[indices[i]];
    }
    const auto end = Clock::now();

    g_sink = sum;
    return std::chrono::duration_cast<Nanoseconds>(end - start).count();
}

long long runBestOfSequential(const std::vector<int>& data, int repeats) {
    long long best = std::numeric_limits<long long>::max();
    for (int r = 0; r < repeats; ++r) {
        const long long t = measureSequential(data);
        best = std::min(best, t);
    }
    return best;
}

long long runBestOfRandom(const std::vector<int>& data, int repeats, std::mt19937_64& rng) {
    long long best = std::numeric_limits<long long>::max();
    for (int r = 0; r < repeats; ++r) {
        const long long t = measureRandom(data, rng);
        best = std::min(best, t);
    }
    return best;
}

long long runBestOfRandomWithIndexArray(
    const std::vector<int>& data,
    int repeats,
    std::mt19937_64& rng
) {
    long long best = std::numeric_limits<long long>::max();

    for (int r = 0; r < repeats; ++r) {
        auto indices = createRandomIndices(data.size(), rng);
        const long long t = measureRandomWithIndexArray(data, indices);
        best = std::min(best, t);
    }

    return best;
}

ResultRow makeRow(
    const std::string& mode,
    std::size_t sizeBytes,
    std::size_t elements,
    long long totalNs
) {
    ResultRow row{};
    row.mode = mode;
    row.sizeBytes = sizeBytes;
    row.sizeKB = static_cast<double>(sizeBytes) / 1024.0;
    row.sizeMB = static_cast<double>(sizeBytes) / (1024.0 * 1024.0);
    row.elements = elements;
    row.totalNs = totalNs;
    row.nsPerIter = static_cast<double>(totalNs) / static_cast<double>(elements);
    return row;
}

void writeCsv(const std::string& filename, const std::vector<ResultRow>& rows) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot open output CSV file: " + filename);
    }

    out << "mode,size_bytes,size_kb,size_mb,elements,total_ns,ns_per_iter\n";
    out << std::fixed << std::setprecision(6);

    for (const auto& row : rows) {
        out << row.mode << ','
            << row.sizeBytes << ','
            << row.sizeKB << ','
            << row.sizeMB << ','
            << row.elements << ','
            << row.totalNs << ','
            << row.nsPerIter << '\n';
    }
}

int main() {
    try {
        const std::size_t intSize = sizeof(int);
        const int repeats = 5;

        std::cout << "Laboratory work #4: cache and memory latency measurement\n";
        std::cout << "sizeof(int) = " << intSize << " bytes\n";
        std::cout << "Repeats per test = " << repeats << "\n\n";

        if (intSize != 4) {
            std::cout << "Warning: sizeof(int) is not 4 bytes on this platform.\n";
            std::cout << "The methodical guide assumes 4-byte integers.\n\n";
        }

        auto sizes = buildTestSizes();
        std::vector<ResultRow> rows;
        rows.reserve(sizes.size() * 3);

        std::mt19937_64 rng(123456789ULL);

        for (std::size_t idx = 0; idx < sizes.size(); ++idx) {
            const std::size_t sizeBytes = sizes[idx];
            const std::size_t elements = sizeBytes / sizeof(int);

            if (elements == 0) {
                continue;
            }

            auto data = createDataArray(elements);

            const long long seqNs = runBestOfSequential(data, repeats);
            const long long rndNs = runBestOfRandom(data, repeats, rng);
            const long long rndIdxNs = runBestOfRandomWithIndexArray(data, repeats, rng);

            rows.push_back(makeRow("sequential", sizeBytes, elements, seqNs));
            rows.push_back(makeRow("random", sizeBytes, elements, rndNs));
            rows.push_back(makeRow("random_index", sizeBytes, elements, rndIdxNs));

            std::cout << "[" << (idx + 1) << "/" << sizes.size() << "] "
                << "Size = " << std::fixed << std::setprecision(2)
                << static_cast<double>(sizeBytes) / (1024.0 * 1024.0) << " MB"
                << " | seq = " << seqNs
                << " ns | rnd = " << rndNs
                << " ns | rnd_idx = " << rndIdxNs
                << " ns\n";
        }

        writeCsv("results.csv", rows);

        std::cout << "\nDone.\n";
        std::cout << "Results saved to results.csv\n";
        std::cout << "Ignore this value: " << g_sink << "\n";

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
