// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "coeff_order_reference.h"

#include <algorithm>
#include <vector>

#include "vardct_layout.h"

namespace cujpegxl {

CoeffOrderStats coeff_order_stats_reference(const std::array<const std::int32_t*, 3>& ac,
                                            const std::int8_t* acs, std::size_t bw,
                                            std::size_t bh) {
    CoeffOrderStats stats{};
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const int side{acs == nullptr ? 8 : acs[by * bw + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            const int cx{side / 8};
            ++stats.blocks[coeff_order_side_index(side)];
            for (int c{0}; c < 3; ++c) {
                for (int raw{0}; raw < side * side; ++raw) {
                    if (raw / side < cx && raw % side < cx) {
                        continue;
                    }
                    const std::size_t slot{
                        covered_plane_slot(side, bx, by, bw, static_cast<std::size_t>(raw))};
                    if (ac[c][slot] != 0) {
                        ++stats.nonzeros[coeff_order_offset(side, c) + raw];
                    }
                }
            }
        }
    }
    return stats;
}

CoeffOrders coeff_orders_reference(const CoeffOrderStats& stats) {
    struct PosAndCount {
        std::uint32_t pos;
        std::uint32_t count;
    };
    CoeffOrders orders{};
    for (int side : {8, 16, 32}) {
        const std::size_t size{static_cast<std::size_t>(side) * side};
        const std::size_t covered{static_cast<std::size_t>(side / 8) * (side / 8)};
        const std::vector<std::uint32_t> natural{
            natural_coeff_order(static_cast<std::size_t>(side))};
        std::vector<std::uint32_t> natural_index(size);
        for (std::size_t i{0}; i < size; ++i) {
            natural_index[natural[i]] = static_cast<std::uint32_t>(i);
        }
        const std::uint32_t blocks{stats.blocks[coeff_order_side_index(side)]};
        for (int c{0}; c < 3; ++c) {
            const std::size_t offset{coeff_order_offset(side, c)};
            std::vector<PosAndCount> pos_and_count(size);
            for (std::size_t i{0}; i < size; ++i) {
                const std::uint32_t pos{natural[i]};
                pos_and_count[i] = {
                    pos,
                    i < covered ? 0 : coeff_order_key(blocks, stats.nonzeros[offset + pos], side)};
            }
            std::stable_sort(
                pos_and_count.begin(), pos_and_count.end(),
                [](const PosAndCount& a, const PosAndCount& b) { return a.count < b.count; });
            std::vector<std::uint32_t> fenwick(size + 1, 0);
            for (std::size_t k{0}; k < size; ++k) {
                const std::uint32_t s{natural_index[pos_and_count[k].pos]};
                std::uint32_t smaller_used{0};
                for (std::uint32_t i{s + 1}; i != 0; i &= i - 1) {
                    smaller_used += fenwick[i];
                }
                for (std::uint32_t i{s + 1}; i <= size; i += i & (~i + 1)) {
                    ++fenwick[i];
                }
                orders.order[offset + k] = static_cast<std::uint16_t>(pos_and_count[k].pos);
                orders.lehmer[offset + k] = static_cast<std::uint16_t>(s - smaller_used);
            }
        }
    }
    return orders;
}

}  // namespace cujpegxl
