// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string_view>
#include <vector>

namespace benchmark_stats {

double mean_of(const std::vector<double>& samples);
double percentile_of(std::vector<double> samples, double percentile);
void print_stats(const std::vector<double>& times_us, std::string_view codec);

}  // namespace benchmark_stats
