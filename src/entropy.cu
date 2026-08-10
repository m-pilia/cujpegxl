// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy.h"

#include <mutex>
#include <utility>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

namespace cujpegxl {
namespace {

constexpr std::size_t AC_GROUP_BLOCKS = 32;

// libjxl default HybridUintConfig (4, 2, 0), matching the host bitstream writer.
constexpr std::uint32_t SPLIT_EXPONENT = 4;
constexpr std::uint32_t MSB_IN_TOKEN = 2;
constexpr std::uint32_t LSB_IN_TOKEN = 0;

// Physical AC channel order Y, X, B (decoder LoadBlock order) indexing q's
// planes 0=X, 1=Y, 2=B.
__constant__ int CHANNEL_ORDER[3]{1, 0, 2};

// order[k] = libjxl-raster index of the k-th coefficient in DCT8 scan order.
__constant__ std::uint32_t NATURAL_ORDER[64];

void compute_natural_order(std::uint32_t out[64]) {
    constexpr std::size_t dim{8};
    std::size_t cur{1};
    for (std::size_t i{0}; i < dim; ++i) {
        for (std::size_t j{0}; j <= i; ++j) {
            std::size_t x{j};
            std::size_t y{i - j};
            if (i & 1) {
                std::swap(x, y);
            }
            const std::size_t val{(x < 1 && y < 1) ? 0 : cur++};
            out[val] = static_cast<std::uint32_t>(y * dim + x);
        }
    }
    for (std::size_t ip{dim - 1}; ip > 0; --ip) {
        const std::size_t i{ip - 1};
        for (std::size_t j{0}; j <= i; ++j) {
            std::size_t x{dim - 1 - (i - j)};
            std::size_t y{dim - 1 - j};
            if (i & 1) {
                std::swap(x, y);
            }
            out[cur++] = static_cast<std::uint32_t>(y * dim + x);
        }
    }
}

void init_constants() {
    std::uint32_t order[64];
    compute_natural_order(order);
    cudaMemcpyToSymbol(NATURAL_ORDER, order, sizeof(order));
}

void ensure_constants() {
    static std::once_flag flag;
    std::call_once(flag, init_constants);
}

__device__ void hybrid_encode(std::uint32_t value, std::uint32_t& token,
                              std::uint32_t& nbits, std::uint32_t& bits) {
    const std::uint32_t split{1u << SPLIT_EXPONENT};
    if (value < split) {
        token = value;
        nbits = 0;
        bits = 0;
        return;
    }
    std::uint32_t n{31u};
    while ((value & (1u << n)) == 0) {
        --n;
    }
    const std::uint32_t m{value - (1u << n)};
    token = split + ((n - SPLIT_EXPONENT) << (MSB_IN_TOKEN + LSB_IN_TOKEN)) +
            ((m >> (n - MSB_IN_TOKEN)) << LSB_IN_TOKEN) +
            (m & ((1u << LSB_IN_TOKEN) - 1));
    nbits = n - MSB_IN_TOKEN - LSB_IN_TOKEN;
    bits = (value >> LSB_IN_TOKEN) & ((1u << nbits) - 1);
}

__device__ std::uint32_t pack_signed(std::int32_t value) {
    return (static_cast<std::uint32_t>(value) << 1) ^
           static_cast<std::uint32_t>(value >> 31);
}

// LSB-first bit writer matching the host BitWriter, packing into a byte buffer
// from a byte-aligned start.
struct DeviceBitWriter {
    std::uint8_t* dst;
    std::size_t byte_pos;
    int bit_pos;
    std::uint8_t cur;

    __device__ void write(std::uint32_t n_bits, std::uint32_t value) {
        while (n_bits > 0) {
            const int take{static_cast<int>(n_bits) < (8 - bit_pos)
                               ? static_cast<int>(n_bits)
                               : (8 - bit_pos)};
            const std::uint32_t mask{(1u << take) - 1};
            cur |= static_cast<std::uint8_t>((value & mask) << bit_pos);
            value >>= take;
            n_bits -= take;
            bit_pos += take;
            if (bit_pos == 8) {
                dst[byte_pos] = cur;
                ++byte_pos;
                cur = 0;
                bit_pos = 0;
            }
        }
    }

    __device__ void flush() {
        if (bit_pos != 0) {
            dst[byte_pos] = cur;
            ++byte_pos;
            cur = 0;
            bit_pos = 0;
        }
    }
};

struct GroupExtent {
    std::size_t bx0;
    std::size_t by0;
    std::size_t gbw;
    std::size_t gbh;
};

__device__ GroupExtent group_extent(std::size_t g, std::size_t bw, std::size_t bh,
                                    std::size_t xg) {
    const std::size_t gx{g % xg};
    const std::size_t gy{g / xg};
    GroupExtent e{};
    e.bx0 = gx * AC_GROUP_BLOCKS;
    e.by0 = gy * AC_GROUP_BLOCKS;
    e.gbw = min(AC_GROUP_BLOCKS, bw - e.bx0);
    e.gbh = min(AC_GROUP_BLOCKS, bh - e.by0);
    return e;
}

// Iterates one block-channel's tokens, invoking `emit(symbol, nbits, bits)` in
// the exact order the decoder reads them.
template <typename Emit>
__device__ void for_each_token(const std::int32_t* blk, Emit emit) {
    std::uint32_t nzeros{0};
    for (int k{1}; k < 64; ++k) {
        if (blk[NATURAL_ORDER[k]] != 0) {
            ++nzeros;
        }
    }
    std::uint32_t symbol{};
    std::uint32_t nbits{};
    std::uint32_t bits{};
    hybrid_encode(nzeros, symbol, nbits, bits);
    emit(symbol, nbits, bits);

    std::uint32_t remaining{nzeros};
    for (int k{1}; k < 64 && remaining > 0; ++k) {
        const std::int32_t v{blk[NATURAL_ORDER[k]]};
        hybrid_encode(pack_signed(v), symbol, nbits, bits);
        emit(symbol, nbits, bits);
        if (v != 0) {
            --remaining;
        }
    }
}

__global__ void histogram_kernel(const std::int32_t* q, std::size_t bw, std::size_t bh,
                                 std::size_t plane, std::uint32_t* histogram) {
    const std::size_t idx{blockIdx.x * blockDim.x + threadIdx.x};
    const std::size_t total{3 * bw * bh};
    if (idx >= total) {
        return;
    }
    const std::size_t block{idx / 3};
    const int p{static_cast<int>(idx % 3)};
    const int c{CHANNEL_ORDER[p]};
    const std::int32_t* blk{q + static_cast<std::size_t>(c) * plane + block * 64};
    for_each_token(blk, [&](std::uint32_t symbol, std::uint32_t, std::uint32_t) {
        atomicAdd(&histogram[symbol], 1u);
    });
}

__global__ void group_size_kernel(const std::int32_t* q, std::size_t bw, std::size_t bh,
                                  std::size_t plane, std::size_t xg,
                                  std::size_t num_groups, const std::uint8_t* depth,
                                  std::uint32_t* group_sizes) {
    const std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
    if (g >= num_groups) {
        return;
    }
    const GroupExtent e{group_extent(g, bw, bh, xg)};
    std::size_t nbits_total{0};
    for (std::size_t by{0}; by < e.gbh; ++by) {
        for (std::size_t bx{0}; bx < e.gbw; ++bx) {
            const std::size_t block{(e.by0 + by) * bw + (e.bx0 + bx)};
            for (int p{0}; p < 3; ++p) {
                const int c{CHANNEL_ORDER[p]};
                const std::int32_t* blk{q + static_cast<std::size_t>(c) * plane +
                                        block * 64};
                for_each_token(blk, [&](std::uint32_t symbol, std::uint32_t nbits,
                                        std::uint32_t) {
                    nbits_total += depth[symbol] + nbits;
                });
            }
        }
    }
    group_sizes[g] = static_cast<std::uint32_t>((nbits_total + 7) / 8);
}

__global__ void group_emit_kernel(const std::int32_t* q, std::size_t bw, std::size_t bh,
                                  std::size_t plane, std::size_t xg,
                                  std::size_t num_groups, const std::uint8_t* depth,
                                  const std::uint16_t* bits_table, std::uint8_t* out,
                                  const std::uint32_t* group_offsets) {
    const std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
    if (g >= num_groups) {
        return;
    }
    const GroupExtent e{group_extent(g, bw, bh, xg)};
    DeviceBitWriter w{out + group_offsets[g], 0, 0, 0};
    for (std::size_t by{0}; by < e.gbh; ++by) {
        for (std::size_t bx{0}; bx < e.gbw; ++bx) {
            const std::size_t block{(e.by0 + by) * bw + (e.bx0 + bx)};
            for (int p{0}; p < 3; ++p) {
                const int c{CHANNEL_ORDER[p]};
                const std::int32_t* blk{q + static_cast<std::size_t>(c) * plane +
                                        block * 64};
                for_each_token(blk, [&](std::uint32_t symbol, std::uint32_t nbits,
                                        std::uint32_t raw) {
                    w.write(depth[symbol], bits_table[symbol]);
                    if (nbits) {
                        w.write(nbits, raw);
                    }
                });
            }
        }
    }
    w.flush();
}

}  // namespace

std::size_t ac_num_groups(std::size_t width, std::size_t height) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xg{(bw + AC_GROUP_BLOCKS - 1) / AC_GROUP_BLOCKS};
    const std::size_t yg{(bh + AC_GROUP_BLOCKS - 1) / AC_GROUP_BLOCKS};
    return xg * yg;
}

bool ac_build_histogram(const std::int32_t* q, std::size_t width, std::size_t height,
                        std::uint32_t* histogram) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t plane{width * height};
    if (cudaMemset(histogram, 0, AC_HISTOGRAM_SIZE * sizeof(std::uint32_t)) !=
        cudaSuccess) {
        return false;
    }
    const std::size_t total{3 * bw * bh};
    const unsigned int threads{256};
    const unsigned int blocks{static_cast<unsigned int>((total + threads - 1) / threads)};
    histogram_kernel<<<blocks, threads>>>(q, bw, bh, plane, histogram);
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

bool ac_encode_groups(const std::int32_t* q, std::size_t width, std::size_t height,
                      const std::uint8_t* depth, const std::uint16_t* bits,
                      std::size_t alphabet_size, std::uint8_t* out,
                      std::size_t out_capacity, std::uint32_t* group_sizes,
                      std::uint32_t* group_offsets, std::size_t* total_bytes) {
    ensure_constants();
    (void)alphabet_size;
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t plane{width * height};
    const std::size_t xg{(bw + AC_GROUP_BLOCKS - 1) / AC_GROUP_BLOCKS};
    const std::size_t num_groups{ac_num_groups(width, height)};

    const unsigned int threads{128};
    const unsigned int blocks{static_cast<unsigned int>((num_groups + threads - 1) / threads)};
    group_size_kernel<<<blocks, threads>>>(q, bw, bh, plane, xg, num_groups, depth,
                                           group_sizes);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        return false;
    }

    void* d_temp{nullptr};
    std::size_t temp_bytes{0};
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, group_sizes, group_offsets,
                                  static_cast<int>(num_groups));
    if (cudaMalloc(&d_temp, temp_bytes) != cudaSuccess) {
        return false;
    }
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, group_sizes, group_offsets,
                                  static_cast<int>(num_groups));
    const cudaError_t scan_status{cudaDeviceSynchronize()};
    cudaFree(d_temp);
    if (scan_status != cudaSuccess) {
        return false;
    }

    std::uint32_t last_offset{0};
    std::uint32_t last_size{0};
    if (cudaMemcpy(&last_offset, group_offsets + (num_groups - 1), sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&last_size, group_sizes + (num_groups - 1), sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    *total_bytes = static_cast<std::size_t>(last_offset) + last_size;
    if (*total_bytes > out_capacity) {
        return false;
    }

    group_emit_kernel<<<blocks, threads>>>(q, bw, bh, plane, xg, num_groups, depth, bits,
                                           out, group_offsets);
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

}  // namespace cujpegxl
