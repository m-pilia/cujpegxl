// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "tools/benchmark/benchmark_stats.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace benchmark_stats {

double mean_of(const std::vector<double>& samples) {
    assert(!samples.empty());
    return std::accumulate(samples.begin(), samples.end(), 0.0) /
           static_cast<double>(samples.size());
}

double percentile_of(std::vector<double> samples, double percentile) {
    assert(!samples.empty());
    assert(percentile >= 0.0 && percentile <= 1.0);
    std::sort(samples.begin(), samples.end());
    const auto index{static_cast<std::size_t>(std::llround(percentile * (samples.size() - 1)))};
    return samples[index];
}

void print_stats(const std::vector<double>& times_us, std::string_view codec) {
    const double total_us{std::accumulate(times_us.begin(), times_us.end(), 0.0)};
    const double mean_us{mean_of(times_us)};
    const double fps{mean_us > 0.0 ? 1.0e6 / mean_us : 0.0};
    const double mean{mean_of(times_us)};
    double squared_error{0.0};
    for (const double time_us : times_us) {
        const double error{time_us - mean};
        squared_error += error * error;
    }
    const double stddev_us{
        times_us.size() < 2 ? 0.0
                            : std::sqrt(squared_error / static_cast<double>(times_us.size() - 1))};
    std::printf("\nresults (%.*s, microseconds):\n", static_cast<int>(codec.size()), codec.data());
    std::printf("  n:          %zu\n", times_us.size());
    std::printf("  total:      %.0f\n", total_us);
    std::printf("  mean:       %.0f   (%.2f fps)\n", mean_us, fps);
    std::printf("  stddev:     %.0f\n", stddev_us);
    std::printf("  min:        %.0f\n", *std::min_element(times_us.begin(), times_us.end()));
    std::printf("  p50:        %.0f\n", percentile_of(times_us, 0.50));
    std::printf("  p99:        %.0f\n", percentile_of(times_us, 0.99));
    std::printf("  max:        %.0f\n", *std::max_element(times_us.begin(), times_us.end()));
}

}  // namespace benchmark_stats
