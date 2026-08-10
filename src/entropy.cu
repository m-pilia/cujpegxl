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
constexpr std::size_t DC_GROUP_BLOCKS = 256;

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

__device__ GroupExtent dc_group_extent(std::size_t g, std::size_t bw, std::size_t bh,
                                       std::size_t xdg) {
    const std::size_t gx{g % xdg};
    const std::size_t gy{g / xdg};
    GroupExtent e{};
    e.bx0 = gx * DC_GROUP_BLOCKS;
    e.by0 = gy * DC_GROUP_BLOCKS;
    e.gbw = min(DC_GROUP_BLOCKS, bw - e.bx0);
    e.gbh = min(DC_GROUP_BLOCKS, bh - e.by0);
    return e;
}

constexpr int DC_EMIT_THREADS = 256;

// A DcGroup is emitted cooperatively by one thread block: its section is an
// ordered sequence of `m` items (blob_pre, then the DC tokens, then blob_mid,
// then the AcMetadata tokens), each contributing a known number of bits. Threads
// own contiguous runs of items, a block-wide scan of per-run bit counts gives
// each run's bit offset, and each item is written at its offset via atomicOr so
// runs that share a byte compose without a race. DcCtx caches the per-group
// geometry and the AcMetadata channel-sample structure (pre_zeros leading zero
// samples = YtoX + YtoB + ACS+QF row 0, then qf_count quant-field samples = row
// 1, then post_zeros zero samples = EPF), matching the host reference.
struct DcCtx {
    const std::int32_t* q;
    std::size_t bw;
    std::size_t plane;
    std::size_t bx0, by0, gbw, gbh;
    const std::uint8_t* dcd;
    const std::uint16_t* dcb;
    const std::uint8_t* amd;
    const std::uint16_t* amb;
    const std::uint8_t* pre;
    std::uint32_t pre_bits;
    const std::uint8_t* mid;
    std::uint32_t mid_bits;
    std::uint32_t qsym, qrn, qraw;
    std::size_t n;  // DC samples per channel = gbw * gbh
    std::size_t pre_zeros, qf_count, post_zeros;
    std::size_t m;  // total items
};

__device__ DcCtx build_dc_ctx(std::size_t g, const std::int32_t* q, std::size_t bw,
                              std::size_t bh, std::size_t plane, std::size_t xdg,
                              std::uint32_t raw_quant_field, const std::uint8_t* dc_depth,
                              const std::uint16_t* dc_bits, const std::uint8_t* acmeta_depth,
                              const std::uint16_t* acmeta_bits, const std::uint8_t* blob_pre,
                              const std::uint32_t* blob_pre_off,
                              const std::uint32_t* blob_pre_bits, const std::uint8_t* blob_mid,
                              const std::uint32_t* blob_mid_off,
                              const std::uint32_t* blob_mid_bits) {
    const GroupExtent e{dc_group_extent(g, bw, bh, xdg)};
    DcCtx c{};
    c.q = q;
    c.bw = bw;
    c.plane = plane;
    c.bx0 = e.bx0;
    c.by0 = e.by0;
    c.gbw = e.gbw;
    c.gbh = e.gbh;
    c.dcd = dc_depth + g * AC_HISTOGRAM_SIZE;
    c.dcb = dc_bits + g * AC_HISTOGRAM_SIZE;
    c.amd = acmeta_depth + g * AC_HISTOGRAM_SIZE;
    c.amb = acmeta_bits + g * AC_HISTOGRAM_SIZE;
    c.pre = blob_pre + blob_pre_off[g];
    c.pre_bits = blob_pre_bits[g];
    c.mid = blob_mid + blob_mid_off[g];
    c.mid_bits = blob_mid_bits[g];
    hybrid_encode(pack_signed(static_cast<std::int32_t>(raw_quant_field) - 1), c.qsym, c.qrn,
                  c.qraw);
    c.n = e.gbw * e.gbh;
    const std::size_t cw{(e.gbw + 7) / 8};
    const std::size_t ch{(e.gbh + 7) / 8};
    c.pre_zeros = 2 * cw * ch + c.n;
    c.qf_count = c.n;
    c.post_zeros = c.n;
    c.m = 3 * c.n + 2 + c.pre_zeros + c.qf_count + c.post_zeros;
    return c;
}

// Bit count of item k in the section's item ordering.
__device__ std::uint32_t dc_item_bits(const DcCtx& c, std::size_t k) {
    if (k == 0) {
        return c.pre_bits;
    }
    if (k < 1 + 3 * c.n) {
        const std::size_t i{k - 1};
        const std::size_t bp{i % c.n};
        const int ch{CHANNEL_ORDER[i / c.n]};
        const std::size_t block{(c.by0 + bp / c.gbw) * c.bw + (c.bx0 + bp % c.gbw)};
        const std::int32_t dc{c.q[static_cast<std::size_t>(ch) * c.plane + block * 64]};
        std::uint32_t sym{}, rn{}, rw{};
        hybrid_encode(pack_signed(dc), sym, rn, rw);
        return c.dcd[sym] + rn;
    }
    if (k == 1 + 3 * c.n) {
        return c.mid_bits;
    }
    const std::size_t j{k - (3 * c.n + 2)};
    if (j < c.pre_zeros || j >= c.pre_zeros + c.qf_count) {
        return c.amd[0];
    }
    return c.amd[c.qsym] + c.qrn;
}

// LSB-first write of `n_bits` of `value` at absolute bit position `pos` into the
// 32-bit word buffer, via atomicOr so concurrent writers of disjoint bit ranges
// (adjacent runs sharing a byte) compose. Requires n_bits <= 31 and a
// zero-initialized buffer.
__device__ void atomic_put_bits(unsigned int* words, unsigned long long pos,
                                std::uint32_t n_bits, std::uint32_t value) {
    while (n_bits > 0) {
        const unsigned long long word{pos >> 5};
        const std::uint32_t bit_in_word{static_cast<std::uint32_t>(pos & 31)};
        const std::uint32_t take{min(n_bits, 32u - bit_in_word)};
        const std::uint32_t chunk{value & ((1u << take) - 1u)};
        atomicOr(&words[word], chunk << bit_in_word);
        value >>= take;
        n_bits -= take;
        pos += take;
    }
}

struct AtomicBitWriter {
    unsigned int* words;
    unsigned long long pos;

    __device__ void put(std::uint32_t n_bits, std::uint32_t value) {
        atomic_put_bits(words, pos, n_bits, value);
        pos += n_bits;
    }

    __device__ void put_blob(const std::uint8_t* blob, std::uint32_t n_bits) {
        const std::uint32_t full{n_bits / 8};
        for (std::uint32_t i{0}; i < full; ++i) {
            put(8, blob[i]);
        }
        const std::uint32_t rem{n_bits & 7};
        if (rem) {
            put(rem, blob[full] & ((1u << rem) - 1u));
        }
    }
};

// Emits item k at the writer's current position.
__device__ void dc_item_emit(const DcCtx& c, std::size_t k, AtomicBitWriter& w) {
    if (k == 0) {
        w.put_blob(c.pre, c.pre_bits);
        return;
    }
    if (k < 1 + 3 * c.n) {
        const std::size_t i{k - 1};
        const std::size_t bp{i % c.n};
        const int ch{CHANNEL_ORDER[i / c.n]};
        const std::size_t block{(c.by0 + bp / c.gbw) * c.bw + (c.bx0 + bp % c.gbw)};
        const std::int32_t dc{c.q[static_cast<std::size_t>(ch) * c.plane + block * 64]};
        std::uint32_t sym{}, rn{}, rw{};
        hybrid_encode(pack_signed(dc), sym, rn, rw);
        w.put(c.dcd[sym], c.dcb[sym]);
        if (rn) {
            w.put(rn, rw);
        }
        return;
    }
    if (k == 1 + 3 * c.n) {
        w.put_blob(c.mid, c.mid_bits);
        return;
    }
    const std::size_t j{k - (3 * c.n + 2)};
    if (j < c.pre_zeros || j >= c.pre_zeros + c.qf_count) {
        w.put(c.amd[0], c.amb[0]);
    } else {
        w.put(c.amd[c.qsym], c.amb[c.qsym]);
        if (c.qrn) {
            w.put(c.qrn, c.qraw);
        }
    }
}

// Block-wide exclusive scan of `val` over `sh` (blockDim entries); returns this
// thread's exclusive prefix and writes the inclusive total.
__device__ unsigned long long block_scan_exclusive(unsigned long long val,
                                                   unsigned long long* sh,
                                                   unsigned long long& total) {
    const unsigned int tid{threadIdx.x};
    sh[tid] = val;
    __syncthreads();
    for (unsigned int offset{1}; offset < blockDim.x; offset <<= 1) {
        unsigned long long add{0};
        if (tid >= offset) {
            add = sh[tid - offset];
        }
        __syncthreads();
        sh[tid] += add;
        __syncthreads();
    }
    total = sh[blockDim.x - 1];
    const unsigned long long inclusive{sh[tid]};
    __syncthreads();
    return inclusive - val;
}

__global__ void dc_histogram_kernel(const std::int32_t* q, std::size_t bw, std::size_t bh,
                                    std::size_t plane, std::size_t xdg,
                                    std::uint32_t* histograms) {
    const std::size_t idx{blockIdx.x * blockDim.x + threadIdx.x};
    const std::size_t total{3 * bw * bh};
    if (idx >= total) {
        return;
    }
    const std::size_t block{idx / 3};
    const int p{static_cast<int>(idx % 3)};
    const int c{CHANNEL_ORDER[p]};
    const std::size_t bx{block % bw};
    const std::size_t by{block / bw};
    const std::size_t g{(by / DC_GROUP_BLOCKS) * xdg + (bx / DC_GROUP_BLOCKS)};
    const std::int32_t dc{q[static_cast<std::size_t>(c) * plane + block * 64]};
    std::uint32_t symbol{};
    std::uint32_t nbits{};
    std::uint32_t bits{};
    hybrid_encode(pack_signed(dc), symbol, nbits, bits);
    atomicAdd(&histograms[g * AC_HISTOGRAM_SIZE + symbol], 1u);
}

__global__ void dc_group_size_kernel(const std::int32_t* q, std::size_t bw, std::size_t bh,
                                     std::size_t plane, std::size_t xdg,
                                     std::size_t num_groups, std::uint32_t raw_quant_field,
                                     const std::uint8_t* dc_depth, const std::uint16_t* dc_bits,
                                     const std::uint8_t* acmeta_depth,
                                     const std::uint16_t* acmeta_bits,
                                     const std::uint8_t* blob_pre,
                                     const std::uint32_t* blob_pre_off,
                                     const std::uint32_t* blob_pre_bits,
                                     const std::uint8_t* blob_mid,
                                     const std::uint32_t* blob_mid_off,
                                     const std::uint32_t* blob_mid_bits,
                                     std::uint32_t* group_sizes) {
    const std::size_t g{blockIdx.x};
    if (g >= num_groups) {
        return;
    }
    const DcCtx c{build_dc_ctx(g, q, bw, bh, plane, xdg, raw_quant_field, dc_depth, dc_bits,
                               acmeta_depth, acmeta_bits, blob_pre, blob_pre_off, blob_pre_bits,
                               blob_mid, blob_mid_off, blob_mid_bits)};
    const std::size_t r{(c.m + blockDim.x - 1) / blockDim.x};
    const std::size_t k0{min(static_cast<std::size_t>(threadIdx.x) * r, c.m)};
    const std::size_t k1{min(k0 + r, c.m)};
    unsigned long long run_bits{0};
    for (std::size_t k{k0}; k < k1; ++k) {
        run_bits += dc_item_bits(c, k);
    }

    __shared__ unsigned long long sh[DC_EMIT_THREADS];
    unsigned long long total{0};
    block_scan_exclusive(run_bits, sh, total);
    if (threadIdx.x == 0) {
        group_sizes[g] = static_cast<std::uint32_t>((total + 7) / 8);
    }
}

__global__ void dc_group_emit_kernel(const std::int32_t* q, std::size_t bw, std::size_t bh,
                                     std::size_t plane, std::size_t xdg,
                                     std::size_t num_groups, std::uint32_t raw_quant_field,
                                     const std::uint8_t* dc_depth, const std::uint16_t* dc_bits,
                                     const std::uint8_t* acmeta_depth,
                                     const std::uint16_t* acmeta_bits,
                                     const std::uint8_t* blob_pre,
                                     const std::uint32_t* blob_pre_off,
                                     const std::uint32_t* blob_pre_bits,
                                     const std::uint8_t* blob_mid,
                                     const std::uint32_t* blob_mid_off,
                                     const std::uint32_t* blob_mid_bits, std::uint8_t* out,
                                     const std::uint32_t* group_offsets) {
    const std::size_t g{blockIdx.x};
    if (g >= num_groups) {
        return;
    }
    const DcCtx c{build_dc_ctx(g, q, bw, bh, plane, xdg, raw_quant_field, dc_depth, dc_bits,
                               acmeta_depth, acmeta_bits, blob_pre, blob_pre_off, blob_pre_bits,
                               blob_mid, blob_mid_off, blob_mid_bits)};
    const std::size_t r{(c.m + blockDim.x - 1) / blockDim.x};
    const std::size_t k0{min(static_cast<std::size_t>(threadIdx.x) * r, c.m)};
    const std::size_t k1{min(k0 + r, c.m)};
    unsigned long long run_bits{0};
    for (std::size_t k{k0}; k < k1; ++k) {
        run_bits += dc_item_bits(c, k);
    }

    __shared__ unsigned long long sh[DC_EMIT_THREADS];
    unsigned long long total{0};
    const unsigned long long run_off{block_scan_exclusive(run_bits, sh, total)};

    AtomicBitWriter w{reinterpret_cast<unsigned int*>(out),
                      (static_cast<unsigned long long>(group_offsets[g]) << 3) + run_off};
    for (std::size_t k{k0}; k < k1; ++k) {
        dc_item_emit(c, k, w);
    }
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

std::size_t dc_num_groups(std::size_t width, std::size_t height) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t ydg{(bh + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    return xdg * ydg;
}

bool dc_build_histograms(const std::int32_t* q, std::size_t width, std::size_t height,
                         std::uint32_t* histograms) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t plane{width * height};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t num_groups{dc_num_groups(width, height)};
    if (cudaMemset(histograms, 0,
                   num_groups * AC_HISTOGRAM_SIZE * sizeof(std::uint32_t)) != cudaSuccess) {
        return false;
    }
    const std::size_t total{3 * bw * bh};
    const unsigned int threads{256};
    const unsigned int blocks{static_cast<unsigned int>((total + threads - 1) / threads)};
    dc_histogram_kernel<<<blocks, threads>>>(q, bw, bh, plane, xdg, histograms);
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

bool dc_encode_groups(const std::int32_t* q, std::size_t width, std::size_t height,
                      std::uint32_t raw_quant_field, const std::uint8_t* dc_depth,
                      const std::uint16_t* dc_bits, const std::uint8_t* acmeta_depth,
                      const std::uint16_t* acmeta_bits, const std::uint8_t* blob_pre,
                      const std::uint32_t* blob_pre_off, const std::uint32_t* blob_pre_bits,
                      const std::uint8_t* blob_mid, const std::uint32_t* blob_mid_off,
                      const std::uint32_t* blob_mid_bits, std::uint8_t* out,
                      std::size_t out_capacity, std::uint32_t* group_sizes,
                      std::uint32_t* group_offsets, std::size_t* total_bytes) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t plane{width * height};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t num_groups{dc_num_groups(width, height)};

    const unsigned int grid{static_cast<unsigned int>(num_groups)};
    dc_group_size_kernel<<<grid, DC_EMIT_THREADS>>>(q, bw, bh, plane, xdg, num_groups,
                                                    raw_quant_field, dc_depth, dc_bits,
                                                    acmeta_depth, acmeta_bits, blob_pre,
                                                    blob_pre_off, blob_pre_bits, blob_mid,
                                                    blob_mid_off, blob_mid_bits, group_sizes);
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

    // The emit kernel composes bits with atomicOr (runs sharing a byte), so the
    // section region must start zeroed. Round up to the 32-bit word the last bits
    // land in; the wrapper's capacity has slack for the padding.
    const std::size_t zero_bytes{(*total_bytes + 3) & ~std::size_t{3}};
    if (cudaMemset(out, 0, zero_bytes) != cudaSuccess) {
        return false;
    }

    dc_group_emit_kernel<<<grid, DC_EMIT_THREADS>>>(q, bw, bh, plane, xdg, num_groups,
                                                    raw_quant_field, dc_depth, dc_bits,
                                                    acmeta_depth, acmeta_bits, blob_pre,
                                                    blob_pre_off, blob_pre_bits, blob_mid,
                                                    blob_mid_off, blob_mid_bits, out,
                                                    group_offsets);
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

}  // namespace cujpegxl
