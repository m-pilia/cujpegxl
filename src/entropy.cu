// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy.h"

#include <mutex>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include "ac_context.h"
#include "coeff_order.h"
#include "dc_predict.h"
#include "vardct_layout.h"

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

// order[k] = libjxl-raster index of the k-th coefficient in the square DCT scan
// order, per transform size. NATURAL_ORDER is the 8x8 table; the DCT16/DCT32
// tables drive the larger blocks of the mixed-block tokenizer.
__constant__ std::uint32_t NATURAL_ORDER[64];
__constant__ std::uint32_t NATURAL_ORDER_16[256];
__constant__ std::uint32_t NATURAL_ORDER_32[1024];

__device__ const std::uint32_t* natural_order_dev(int side) {
    return side == 16 ? NATURAL_ORDER_16 : (side == 32 ? NATURAL_ORDER_32 : NATURAL_ORDER);
}

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
    const std::vector<std::uint32_t> o16{natural_coeff_order(16)};
    const std::vector<std::uint32_t> o32{natural_coeff_order(32)};
    cudaMemcpyToSymbol(NATURAL_ORDER_16, o16.data(), o16.size() * sizeof(std::uint32_t));
    cudaMemcpyToSymbol(NATURAL_ORDER_32, o32.data(), o32.size() * sizeof(std::uint32_t));
}

void ensure_constants() {
    static std::once_flag flag;
    std::call_once(flag, init_constants);
}

__device__ void hybrid_encode(std::uint32_t value, std::uint32_t& token, std::uint32_t& nbits,
                              std::uint32_t& bits) {
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
            ((m >> (n - MSB_IN_TOKEN)) << LSB_IN_TOKEN) + (m & ((1u << LSB_IN_TOKEN) - 1));
    nbits = n - MSB_IN_TOKEN - LSB_IN_TOKEN;
    bits = (value >> LSB_IN_TOKEN) & ((1u << nbits) - 1);
}

__device__ std::uint32_t pack_signed(std::int32_t value) {
    return (static_cast<std::uint32_t>(value) << 1) ^ static_cast<std::uint32_t>(value >> 31);
}

// Gradient prediction for the DC sample of channel plane `ch` at block (bx, by),
// with the neighbors taken within the block's DC group (origin bx0, by0). DC is
// coded losslessly, so the original neighbor values reproduce the decoder's, and
// every residual is independent (no reconstruction dependency).
__device__ std::int32_t dc_group_gradient(const std::int32_t* dc, int ch, std::size_t nblocks,
                                          std::size_t bw, std::size_t bx, std::size_t by,
                                          std::size_t bx0, std::size_t by0) {
    const std::size_t base{static_cast<std::size_t>(ch) * nblocks};
    const bool has_left{bx > bx0};
    const bool has_top{by > by0};
    const std::int32_t w_val{has_left ? dc[base + by * bw + (bx - 1)] : 0};
    const std::int32_t n_val{has_top ? dc[base + (by - 1) * bw + bx] : 0};
    const std::int32_t nw_val{(has_left && has_top) ? dc[base + (by - 1) * bw + (bx - 1)] : 0};
    return gradient_predict(has_left, has_top, w_val, n_val, nw_val);
}

// libjxl AcStrategyType raw value for a square block side (DCT=0, DCT16X16=4,
// DCT32X32=5).
__device__ std::int32_t raw_strategy_dev(int side) {
    return side == 16 ? 4 : (side == 32 ? 5 : 0);
}

struct GroupExtent {
    std::size_t bx0;
    std::size_t by0;
    std::size_t gbw;
    std::size_t gbh;
};

__device__ GroupExtent group_extent(std::size_t g, std::size_t bw, std::size_t bh, std::size_t xg) {
    const std::size_t gx{g % xg};
    const std::size_t gy{g / xg};
    GroupExtent e{};
    e.bx0 = gx * AC_GROUP_BLOCKS;
    e.by0 = gy * AC_GROUP_BLOCKS;
    e.gbw = min(AC_GROUP_BLOCKS, bw - e.bx0);
    e.gbh = min(AC_GROUP_BLOCKS, bh - e.by0);
    return e;
}

// Per-cluster histogram span: AC_NUM_CLUSTERS histograms of AC_HISTOGRAM_SIZE.
constexpr std::size_t AC_CLUSTER_HIST = AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE;

// Group-local (AC group = 32 blocks) neighbor-predicted non-zero count for block
// (bx, by) of physical channel `c`, read from the normalized non-zero grid
// (3 * nblocks int32, covered positions filled).
__device__ std::uint32_t ac_predicted_nzeros_dev(const std::int32_t* nz_grid, int c,
                                                 std::size_t nblocks, std::size_t bw,
                                                 std::size_t bx, std::size_t by) {
    const bool has_left{(bx % AC_GROUP_BLOCKS) > 0};
    const bool has_top{(by % AC_GROUP_BLOCKS) > 0};
    const std::size_t base{static_cast<std::size_t>(c) * nblocks};
    const std::int32_t topv{has_top ? nz_grid[base + (by - 1) * bw + bx] : 0};
    const std::int32_t leftv{has_left ? nz_grid[base + by * bw + (bx - 1)] : 0};
    return ac_predict_nonzeros(has_left, has_top, topv, leftv);
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

// Fills, per DcGroup, the global block indices of its first-blocks in group
// raster order (fb_pos[g*stride + i]) and their count (fb_count[g]). One thread
// per group; a null acs means every block is a DCT8 first-block. The AcMetadata
// emitter reads these to map an ACS/QF channel column (a first-block ordinal)
// back to its block.
__global__ void build_first_blocks_kernel(const std::int8_t* acs, std::size_t bw, std::size_t bh,
                                          std::size_t xdg, std::size_t num_groups,
                                          std::size_t stride, std::size_t* fb_pos,
                                          std::size_t* fb_count) {
    const std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
    if (g >= num_groups) {
        return;
    }
    const GroupExtent e{dc_group_extent(g, bw, bh, xdg)};
    std::size_t cnt{0};
    for (std::size_t by{0}; by < e.gbh; ++by) {
        for (std::size_t bx{0}; bx < e.gbw; ++bx) {
            const std::size_t block{(e.by0 + by) * bw + (e.bx0 + bx)};
            const int side{acs == nullptr ? 8 : acs[block]};
            if (side != ACS_COVERED) {
                fb_pos[g * stride + cnt] = block;
                ++cnt;
            }
        }
    }
    fb_count[g] = cnt;
}

constexpr int DC_EMIT_THREADS = 512;

// A DcGroup holds up to 256x256 blocks, so a single thread block per group leaves
// most of the device idle (a 4K frame has only 4 DcGroups). Each group is instead
// split into DC_CHUNKS contiguous item ranges emitted by separate thread blocks;
// a per-group scan of the chunk bit counts gives each chunk its bit offset within
// the group so the concatenation still reproduces the reference bitstream.
constexpr int DC_CHUNKS = 128;

// A DcGroup is emitted cooperatively by one thread block: its section is an
// ordered sequence of `m` items (blob_pre, then the DC tokens, then blob_mid,
// then the AcMetadata tokens), each contributing a known number of bits. Threads
// own contiguous runs of items, a block-wide scan of per-run bit counts gives
// each run's bit offset, and each item is written at its offset via atomicOr so
// runs that share a byte compose without a race. DcCtx caches the per-group
// geometry and the AcMetadata channel-sample structure: YtoX (cw*ch), YtoB
// (cw*ch), ACS+QF row 0 (count strategy samples), row 1 (count quant-field
// samples), then EPF (n zeros), matching the host reference. `acs`/`ytox_map`/
// `ytob_map` are null for an all-DCT8 base-correlation frame; `fb_pos` lists this
// group's first-block block indices in raster order.
struct DcCtx {
    const std::int32_t* dc;
    std::size_t bw;
    std::size_t nblocks;
    std::size_t bx0, by0, gbw, gbh;
    const std::uint8_t* dcd;
    const std::uint16_t* dcb;
    const std::uint8_t* amd;
    const std::uint16_t* amb;
    const std::uint8_t* pre;
    std::uint32_t pre_bits;
    const std::uint8_t* mid;
    std::uint32_t mid_bits;
    const std::int32_t* qf;
    const std::int8_t* acs;
    const std::int8_t* ytox_map;
    const std::int8_t* ytob_map;
    std::size_t cmw;
    const std::size_t* fb_pos;  // this group's first-block block indices
    std::size_t n;              // DC samples per channel = gbw*gbh (and EPF count)
    std::size_t cw, ch;         // CfL color-tile grid within the group
    std::size_t count;          // first-blocks (ACS+QF columns)
    std::size_t m;              // total items
};

// (symbol, extra bits) of the group's `j`-th AcMetadata sample, in channel/row
// order: YtoX, YtoB, ACS row 0, QF row 1, EPF. Predictor::Zero, so each sample is
// pack_signed of its raw value.
__device__ void dc_acmeta_token(const DcCtx& c, std::size_t j, std::uint32_t& sym,
                                std::uint32_t& rn, std::uint32_t& rw) {
    const std::size_t cwch{c.cw * c.ch};
    std::int32_t v{0};
    if (j < cwch) {
        const std::size_t gt{(c.by0 / 8 + j / c.cw) * c.cmw + (c.bx0 / 8 + j % c.cw)};
        v = c.ytox_map == nullptr ? 0 : c.ytox_map[gt];
    } else if (j < 2 * cwch) {
        const std::size_t jj{j - cwch};
        const std::size_t gt{(c.by0 / 8 + jj / c.cw) * c.cmw + (c.bx0 / 8 + jj % c.cw)};
        v = c.ytob_map == nullptr ? 0 : c.ytob_map[gt];
    } else if (j < 2 * cwch + c.count) {
        const std::size_t block{c.fb_pos[j - 2 * cwch]};
        v = raw_strategy_dev(c.acs == nullptr ? 8 : c.acs[block]);
    } else if (j < 2 * cwch + 2 * c.count) {
        const std::size_t block{c.fb_pos[j - (2 * cwch + c.count)]};
        v = c.qf[block] - 1;
    }
    hybrid_encode(pack_signed(v), sym, rn, rw);
}

__device__ DcCtx build_dc_ctx(std::size_t g, const std::int32_t* dc, std::size_t bw, std::size_t bh,
                              std::size_t nblocks, std::size_t xdg, const std::int32_t* quant_field,
                              const std::int8_t* acs, const std::int8_t* ytox_map,
                              const std::int8_t* ytob_map, std::size_t cmw,
                              const std::size_t* fb_pos, const std::size_t* fb_count,
                              std::size_t fb_stride, const std::uint8_t* dc_depth,
                              const std::uint16_t* dc_bits, const std::uint8_t* acmeta_depth,
                              const std::uint16_t* acmeta_bits, const std::uint8_t* blob_pre,
                              const std::uint32_t* blob_pre_off, const std::uint32_t* blob_pre_bits,
                              const std::uint8_t* blob_mid, const std::uint32_t* blob_mid_off,
                              const std::uint32_t* blob_mid_bits) {
    const GroupExtent e{dc_group_extent(g, bw, bh, xdg)};
    DcCtx c{};
    c.dc = dc;
    c.bw = bw;
    c.nblocks = nblocks;
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
    c.qf = quant_field;
    c.acs = acs;
    c.ytox_map = ytox_map;
    c.ytob_map = ytob_map;
    c.cmw = cmw;
    c.fb_pos = fb_pos + g * fb_stride;
    c.n = e.gbw * e.gbh;
    c.cw = (e.gbw + 7) / 8;
    c.ch = (e.gbh + 7) / 8;
    c.count = fb_count[g];
    c.m = 3 * c.n + 2 + 2 * c.cw * c.ch + 2 * c.count + c.n;
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
        const std::size_t bx{c.bx0 + bp % c.gbw};
        const std::size_t by{c.by0 + bp / c.gbw};
        const std::int32_t dc{c.dc[static_cast<std::size_t>(ch) * c.nblocks + by * c.bw + bx]};
        const std::int32_t pred{dc_group_gradient(c.dc, ch, c.nblocks, c.bw, bx, by, c.bx0, c.by0)};
        std::uint32_t sym{}, rn{}, rw{};
        hybrid_encode(pack_signed(dc - pred), sym, rn, rw);
        return c.dcd[sym] + rn;
    }
    if (k == 1 + 3 * c.n) {
        return c.mid_bits;
    }
    std::uint32_t sym{}, rn{}, rw{};
    dc_acmeta_token(c, k - (3 * c.n + 2), sym, rn, rw);
    return c.amd[sym] + rn;
}

// LSB-first write of `n_bits` of `value` at absolute bit position `pos` into the
// 32-bit word buffer, via atomicOr so concurrent writers of disjoint bit ranges
// (adjacent runs sharing a byte) compose. Requires n_bits <= 31 and a
// zero-initialized buffer.
__device__ void atomic_put_bits(unsigned int* words, unsigned long long pos, std::uint32_t n_bits,
                                std::uint32_t value) {
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
        const std::size_t bx{c.bx0 + bp % c.gbw};
        const std::size_t by{c.by0 + bp / c.gbw};
        const std::int32_t dc{c.dc[static_cast<std::size_t>(ch) * c.nblocks + by * c.bw + bx]};
        const std::int32_t pred{dc_group_gradient(c.dc, ch, c.nblocks, c.bw, bx, by, c.bx0, c.by0)};
        std::uint32_t sym{}, rn{}, rw{};
        hybrid_encode(pack_signed(dc - pred), sym, rn, rw);
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
    std::uint32_t sym{}, rn{}, rw{};
    dc_acmeta_token(c, k - (3 * c.n + 2), sym, rn, rw);
    w.put(c.amd[sym], c.amb[sym]);
    if (rn) {
        w.put(rn, rw);
    }
}

// Block-wide exclusive scan of `val` over `sh` (blockDim entries); returns this
// thread's exclusive prefix and writes the inclusive total.
__device__ unsigned long long block_scan_exclusive(unsigned long long val, unsigned long long* sh,
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

constexpr int AC_EMIT_THREADS = 1024;

__device__ std::uint8_t runtime_ac_cluster(const std::uint8_t* context_map, std::uint32_t context) {
    return context_map == nullptr ? static_cast<std::uint8_t>(ac_cluster(context))
                                  : context_map[context];
}

__global__ void dc_histogram_kernel(const std::int32_t* dc_buf, std::size_t bw, std::size_t bh,
                                    std::size_t nblocks, std::size_t xdg,
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
    const std::size_t bx0{(bx / DC_GROUP_BLOCKS) * DC_GROUP_BLOCKS};
    const std::size_t by0{(by / DC_GROUP_BLOCKS) * DC_GROUP_BLOCKS};
    const std::size_t g{(by / DC_GROUP_BLOCKS) * xdg + (bx / DC_GROUP_BLOCKS)};
    const std::int32_t dc{dc_buf[static_cast<std::size_t>(c) * nblocks + block]};
    const std::int32_t pred{dc_group_gradient(dc_buf, c, nblocks, bw, bx, by, bx0, by0)};
    std::uint32_t symbol{};
    std::uint32_t nbits{};
    std::uint32_t bits{};
    hybrid_encode(pack_signed(dc - pred), symbol, nbits, bits);
    atomicAdd(&histograms[g * AC_HISTOGRAM_SIZE + symbol], 1u);
}

// Accumulates each first-block's ACS strategy token (row 0) and quant-field token
// (row 1) into its DcGroup's AcMetadata histogram.
__global__ void acmeta_acsqf_histogram_kernel(const std::int32_t* quant_field,
                                              const std::int8_t* acs, std::size_t bw,
                                              std::size_t bh, std::size_t xdg,
                                              std::uint32_t* histograms) {
    const std::size_t block{blockIdx.x * blockDim.x + threadIdx.x};
    if (block >= bw * bh) {
        return;
    }
    const int side{acs == nullptr ? 8 : acs[block]};
    if (side == ACS_COVERED) {
        return;
    }
    const std::size_t bx{block % bw};
    const std::size_t by{block / bw};
    const std::size_t g{(by / DC_GROUP_BLOCKS) * xdg + (bx / DC_GROUP_BLOCKS)};
    std::uint32_t symbol{}, nbits{}, bits{};
    hybrid_encode(pack_signed(raw_strategy_dev(side)), symbol, nbits, bits);
    atomicAdd(&histograms[g * AC_HISTOGRAM_SIZE + symbol], 1u);
    hybrid_encode(pack_signed(quant_field[block] - 1), symbol, nbits, bits);
    atomicAdd(&histograms[g * AC_HISTOGRAM_SIZE + symbol], 1u);
}

// Accumulates each 64x64 color tile's YtoX and YtoB CfL tokens into its DcGroup's
// AcMetadata histogram (null maps are the base correlation, symbol 0).
__global__ void acmeta_cfl_histogram_kernel(const std::int8_t* ytox_map,
                                            const std::int8_t* ytob_map, std::size_t cmw,
                                            std::size_t cmh, std::size_t xdg,
                                            std::uint32_t* histograms) {
    const std::size_t tile{blockIdx.x * blockDim.x + threadIdx.x};
    if (tile >= cmw * cmh) {
        return;
    }
    const std::size_t ctx{tile % cmw};
    const std::size_t cty{tile / cmw};
    const std::size_t g{(cty / 32) * xdg + (ctx / 32)};  // 32 color tiles per DC group
    std::uint32_t symbol{}, nbits{}, bits{};
    hybrid_encode(pack_signed(ytox_map == nullptr ? 0 : ytox_map[tile]), symbol, nbits, bits);
    atomicAdd(&histograms[g * AC_HISTOGRAM_SIZE + symbol], 1u);
    hybrid_encode(pack_signed(ytob_map == nullptr ? 0 : ytob_map[tile]), symbol, nbits, bits);
    atomicAdd(&histograms[g * AC_HISTOGRAM_SIZE + symbol], 1u);
}

// Adds each DcGroup's EPF structural zeros (n samples) to symbol 0.
__global__ void acmeta_epf_zeros_kernel(std::size_t bw, std::size_t bh, std::size_t xdg,
                                        std::size_t num_groups, std::uint32_t* histograms) {
    const std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
    if (g >= num_groups) {
        return;
    }
    const GroupExtent e{dc_group_extent(g, bw, bh, xdg)};
    histograms[g * AC_HISTOGRAM_SIZE] += static_cast<std::uint32_t>(e.gbw * e.gbh);
}

// Item range [k_lo, k_hi) of group `c` owned by chunk `cb`.
__device__ void dc_chunk_range(const DcCtx& c, std::size_t cb, std::size_t& k_lo,
                               std::size_t& k_hi) {
    const std::size_t chunk{(c.m + DC_CHUNKS - 1) / DC_CHUNKS};
    k_lo = min(cb * chunk, c.m);
    k_hi = min(k_lo + chunk, c.m);
}

// This thread's item sub-range [a0, a1) within [k_lo, k_hi).
__device__ void dc_thread_range(std::size_t k_lo, std::size_t k_hi, std::size_t& a0,
                                std::size_t& a1) {
    const std::size_t len{k_hi - k_lo};
    const std::size_t r{(len + blockDim.x - 1) / blockDim.x};
    a0 = k_lo + min(static_cast<std::size_t>(threadIdx.x) * r, len);
    a1 = k_lo + min(static_cast<std::size_t>(threadIdx.x) * r + r, len);
}

__global__ void dc_chunk_size_kernel(
    const std::int32_t* dc, std::size_t bw, std::size_t bh, std::size_t nblocks, std::size_t xdg,
    std::size_t num_groups, const std::int32_t* quant_field, const std::int8_t* acs,
    const std::int8_t* ytox_map, const std::int8_t* ytob_map, std::size_t cmw,
    const std::size_t* fb_pos, const std::size_t* fb_count, std::size_t fb_stride,
    const std::uint8_t* dc_depth, const std::uint16_t* dc_bits, const std::uint8_t* acmeta_depth,
    const std::uint16_t* acmeta_bits, const std::uint8_t* blob_pre,
    const std::uint32_t* blob_pre_off, const std::uint32_t* blob_pre_bits,
    const std::uint8_t* blob_mid, const std::uint32_t* blob_mid_off,
    const std::uint32_t* blob_mid_bits, unsigned long long* chunk_bits) {
    const std::size_t g{blockIdx.x};
    if (g >= num_groups) {
        return;
    }
    const DcCtx c{build_dc_ctx(g, dc, bw, bh, nblocks, xdg, quant_field, acs, ytox_map, ytob_map,
                               cmw, fb_pos, fb_count, fb_stride, dc_depth, dc_bits, acmeta_depth,
                               acmeta_bits, blob_pre, blob_pre_off, blob_pre_bits, blob_mid,
                               blob_mid_off, blob_mid_bits)};
    std::size_t k_lo{0};
    std::size_t k_hi{0};
    dc_chunk_range(c, blockIdx.y, k_lo, k_hi);
    std::size_t a0{0};
    std::size_t a1{0};
    dc_thread_range(k_lo, k_hi, a0, a1);
    unsigned long long run_bits{0};
    for (std::size_t k{a0}; k < a1; ++k) {
        run_bits += dc_item_bits(c, k);
    }

    __shared__ unsigned long long sh[DC_EMIT_THREADS];
    unsigned long long total{0};
    block_scan_exclusive(run_bits, sh, total);
    if (threadIdx.x == 0) {
        chunk_bits[g * DC_CHUNKS + blockIdx.y] = total;
    }
}

// Per group: exclusive scan of the DC_CHUNKS chunk bit counts gives each chunk's
// bit offset within the group; the group's total bits give its byte size.
__global__ void dc_group_scan_kernel(std::size_t num_groups, const unsigned long long* chunk_bits,
                                     unsigned long long* chunk_base, std::uint32_t* group_sizes) {
    const std::size_t g{blockIdx.x};
    if (g >= num_groups) {
        return;
    }
    __shared__ unsigned long long sh[DC_CHUNKS];
    unsigned long long total{0};
    const unsigned long long excl{
        block_scan_exclusive(chunk_bits[g * DC_CHUNKS + threadIdx.x], sh, total)};
    chunk_base[g * DC_CHUNKS + threadIdx.x] = excl;
    if (threadIdx.x == 0) {
        group_sizes[g] = static_cast<std::uint32_t>((total + 7) / 8);
    }
}

__global__ void dc_chunk_emit_kernel(
    const std::int32_t* dc, std::size_t bw, std::size_t bh, std::size_t nblocks, std::size_t xdg,
    std::size_t num_groups, const std::int32_t* quant_field, const std::int8_t* acs,
    const std::int8_t* ytox_map, const std::int8_t* ytob_map, std::size_t cmw,
    const std::size_t* fb_pos, const std::size_t* fb_count, std::size_t fb_stride,
    const std::uint8_t* dc_depth, const std::uint16_t* dc_bits, const std::uint8_t* acmeta_depth,
    const std::uint16_t* acmeta_bits, const std::uint8_t* blob_pre,
    const std::uint32_t* blob_pre_off, const std::uint32_t* blob_pre_bits,
    const std::uint8_t* blob_mid, const std::uint32_t* blob_mid_off,
    const std::uint32_t* blob_mid_bits, const unsigned long long* chunk_base, std::uint8_t* out,
    const std::uint32_t* group_offsets) {
    const std::size_t g{blockIdx.x};
    if (g >= num_groups) {
        return;
    }
    const DcCtx c{build_dc_ctx(g, dc, bw, bh, nblocks, xdg, quant_field, acs, ytox_map, ytob_map,
                               cmw, fb_pos, fb_count, fb_stride, dc_depth, dc_bits, acmeta_depth,
                               acmeta_bits, blob_pre, blob_pre_off, blob_pre_bits, blob_mid,
                               blob_mid_off, blob_mid_bits)};
    std::size_t k_lo{0};
    std::size_t k_hi{0};
    dc_chunk_range(c, blockIdx.y, k_lo, k_hi);
    std::size_t a0{0};
    std::size_t a1{0};
    dc_thread_range(k_lo, k_hi, a0, a1);
    unsigned long long run_bits{0};
    for (std::size_t k{a0}; k < a1; ++k) {
        run_bits += dc_item_bits(c, k);
    }

    __shared__ unsigned long long sh[DC_EMIT_THREADS];
    unsigned long long total{0};
    const unsigned long long run_off{block_scan_exclusive(run_bits, sh, total)};

    const unsigned long long base{(static_cast<unsigned long long>(group_offsets[g]) << 3) +
                                  chunk_base[g * DC_CHUNKS + blockIdx.y]};
    AtomicBitWriter w{reinterpret_cast<unsigned int*>(out), base + run_off};
    for (std::size_t k{a0}; k < a1; ++k) {
        dc_item_emit(c, k, w);
    }
}

// --- Mixed-block AC path ------------------------------------------------------
// The coefficient buffer is the covered-block layout (COEFFS_PER_BLOCK slots
// per 8x8 position, per channel plane nblocks*COEFFS_PER_BLOCK); a first-block of
// side N tokenizes order[covered..size) of its channel plane, gathered via
// covered_plane_slot; covered blocks emit nothing. Mirrors the host tokenizer
// (tools/bitstream tokenize_ac_group) so the device stays byte-exact.

// Context-aware mixed-block tokenizer: invokes emit(symbol, nbits, bits, cluster)
// in decoder order, matching the host clustered reference.
template <typename Emit>
__device__ void for_each_token_ctx(const std::int16_t* plane, std::size_t bx, std::size_t by,
                                   int side, std::size_t bw, int c, std::uint32_t predicted,
                                   Emit emit) {
    const int cx{side / 8};
    const std::uint32_t covered{static_cast<std::uint32_t>(cx * cx)};
    const std::uint32_t size{static_cast<std::uint32_t>(side * side)};
    const std::uint32_t l2{side == 16 ? 2u : (side == 32 ? 4u : 0u)};
    const std::uint32_t* order{natural_order_dev(side)};
    const int block_ctx{ac_block_context(c, ac_strategy_order(side))};

    std::uint32_t nzeros{0};
    for (std::uint32_t k{covered}; k < size; ++k) {
        if (plane[covered_plane_slot(side, bx, by, bw, order[k])] != 0) {
            ++nzeros;
        }
    }
    std::uint32_t symbol{}, nbits{}, bits{};
    hybrid_encode(nzeros, symbol, nbits, bits);
    emit(symbol, nbits, bits, ac_nonzero_context(predicted, block_ctx));

    const std::uint32_t histo_off{ac_zero_density_offset(block_ctx)};
    std::uint32_t remaining{nzeros};
    std::uint32_t prev{nzeros > size / 16 ? 0u : 1u};
    for (std::uint32_t k{covered}; k < size && remaining > 0; ++k) {
        const std::int32_t v{plane[covered_plane_slot(side, bx, by, bw, order[k])]};
        const std::uint32_t ctx{histo_off +
                                ac_zero_density_context(remaining, k, covered, l2, prev)};
        hybrid_encode(pack_signed(v), symbol, nbits, bits);
        emit(symbol, nbits, bits, ctx);
        prev = v != 0 ? 1u : 0u;
        remaining -= prev;
    }
}

// Builds the mixed-block normalized non-zero grid: each first-block writes its
// normalized count ((nz + covered - 1) >> log2_covered) to all its covered
// positions, per physical channel.
__global__ void ac_nzeros_grid_kernel(const std::int16_t* ac, const std::int8_t* acs,
                                      std::size_t bw, std::size_t bh, std::size_t nblocks,
                                      std::int32_t* nz_grid) {
    const std::size_t block{blockIdx.x * blockDim.x + threadIdx.x};
    if (block >= bw * bh) {
        return;
    }
    const int side{acs == nullptr ? 8 : acs[block]};
    if (side == ACS_COVERED) {
        return;
    }
    const std::size_t bx{block % bw};
    const std::size_t by{block / bw};
    const int cx{side / 8};
    const std::uint32_t covered{static_cast<std::uint32_t>(cx * cx)};
    const std::uint32_t size{static_cast<std::uint32_t>(side * side)};
    const std::uint32_t l2{side == 16 ? 2u : (side == 32 ? 4u : 0u)};
    const std::uint32_t* order{natural_order_dev(side)};
    for (int p{0}; p < 3; ++p) {
        const int c{CHANNEL_ORDER[p]};
        const std::int16_t* plane{ac + static_cast<std::size_t>(c) * nblocks * COEFFS_PER_BLOCK};
        std::int32_t nz{0};
        for (std::uint32_t k{covered}; k < size; ++k) {
            if (plane[covered_plane_slot(side, bx, by, bw, order[k])] != 0) {
                ++nz;
            }
        }
        const std::int32_t norm{
            static_cast<std::int32_t>((static_cast<std::uint32_t>(nz) + covered - 1) >> l2)};
        for (int dy{0}; dy < cx; ++dy) {
            for (int dx{0}; dx < cx; ++dx) {
                nz_grid[static_cast<std::size_t>(c) * nblocks + (by + dy) * bw + (bx + dx)] = norm;
            }
        }
    }
}

__global__ void histogram_kernel(const std::int16_t* ac, const std::int8_t* acs,
                                 const std::int32_t* nz_grid, std::size_t bw, std::size_t bh,
                                 std::size_t nblocks, std::uint32_t* histogram) {
    __shared__ unsigned int sh[AC_CLUSTER_HIST];
    for (unsigned int i{threadIdx.x}; i < AC_CLUSTER_HIST; i += blockDim.x) {
        sh[i] = 0;
    }
    __syncthreads();

    const std::size_t idx{blockIdx.x * blockDim.x + threadIdx.x};
    const std::size_t total{3 * bw * bh};
    if (idx < total) {
        const std::size_t block{idx / 3};
        const int side{acs == nullptr ? 8 : acs[block]};
        if (side != ACS_COVERED) {
            const int p{static_cast<int>(idx % 3)};
            const int c{CHANNEL_ORDER[p]};
            const std::size_t bx{block % bw};
            const std::size_t by{block / bw};
            const std::int16_t* plane{ac +
                                      static_cast<std::size_t>(c) * nblocks * COEFFS_PER_BLOCK};
            const std::uint32_t predicted{ac_predicted_nzeros_dev(nz_grid, c, nblocks, bw, bx, by)};
            for_each_token_ctx(
                plane, bx, by, side, bw, c, predicted,
                [&](std::uint32_t symbol, std::uint32_t, std::uint32_t, std::uint32_t context) {
                    const int cluster{ac_cluster(context)};
                    atomicAdd(&sh[static_cast<std::size_t>(cluster) * AC_HISTOGRAM_SIZE + symbol],
                              1u);
                });
        }
    }
    __syncthreads();

    for (unsigned int i{threadIdx.x}; i < AC_CLUSTER_HIST; i += blockDim.x) {
        if (sh[i] != 0) {
            atomicAdd(&histogram[i], sh[i]);
        }
    }
}

struct AcGroupCtx {
    const std::int16_t* ac;
    const std::int8_t* acs;
    const std::int32_t* nz_grid;
    std::size_t bw, nblocks;
    std::size_t bx0, by0, gbw;
    const std::uint8_t* depth;
    const std::uint16_t* bits;
    const std::uint8_t* context_map;
    std::size_t n_items;  // gbw * gbh * 3, covered block-channels emit nothing
};

__device__ AcGroupCtx build_ac_ctx(std::size_t g, const std::int16_t* ac, const std::int8_t* acs,
                                   const std::int32_t* nz_grid, std::size_t bw, std::size_t bh,
                                   std::size_t nblocks, std::size_t xg, const std::uint8_t* depth,
                                   const std::uint16_t* bits, const std::uint8_t* context_map) {
    const GroupExtent e{group_extent(g, bw, bh, xg)};
    AcGroupCtx c{};
    c.ac = ac;
    c.acs = acs;
    c.nz_grid = nz_grid;
    c.bw = bw;
    c.nblocks = nblocks;
    c.bx0 = e.bx0;
    c.by0 = e.by0;
    c.gbw = e.gbw;
    c.depth = depth;
    c.bits = bits;
    c.context_map = context_map;
    c.n_items = e.gbw * e.gbh * 3;
    return c;
}

// The (block, channel, predicted nzeros) of item i, and its transform side
// (ACS_COVERED if the item contributes no tokens).
__device__ int ac_item(const AcGroupCtx& c, std::size_t i, std::size_t& bx, std::size_t& by,
                       int& ch, std::uint32_t& predicted, const std::int16_t*& plane) {
    const int p{static_cast<int>(i % 3)};
    const std::size_t bidx{i / 3};
    bx = c.bx0 + bidx % c.gbw;
    by = c.by0 + bidx / c.gbw;
    ch = CHANNEL_ORDER[p];
    const int side{c.acs == nullptr ? 8 : c.acs[by * c.bw + bx]};
    plane = c.ac + static_cast<std::size_t>(ch) * c.nblocks * COEFFS_PER_BLOCK;
    predicted = ac_predicted_nzeros_dev(c.nz_grid, ch, c.nblocks, c.bw, bx, by);
    return side;
}

__device__ std::uint32_t ac_item_bits(const AcGroupCtx& c, std::size_t i) {
    std::size_t bx{}, by{};
    int ch{};
    std::uint32_t predicted{};
    const std::int16_t* plane{};
    const int side{ac_item(c, i, bx, by, ch, predicted, plane)};
    if (side == ACS_COVERED) {
        return 0;
    }
    std::uint32_t total{0};
    for_each_token_ctx(
        plane, bx, by, side, c.bw, ch, predicted,
        [&](std::uint32_t sym, std::uint32_t nbits, std::uint32_t, std::uint32_t context) {
            const std::uint8_t cl{runtime_ac_cluster(c.context_map, context)};
            total += c.depth[static_cast<std::size_t>(cl) * AC_HISTOGRAM_SIZE + sym] + nbits;
        });
    return total;
}

__device__ void ac_item_emit(const AcGroupCtx& c, std::size_t i, AtomicBitWriter& w) {
    std::size_t bx{}, by{};
    int ch{};
    std::uint32_t predicted{};
    const std::int16_t* plane{};
    const int side{ac_item(c, i, bx, by, ch, predicted, plane)};
    if (side == ACS_COVERED) {
        return;
    }
    for_each_token_ctx(
        plane, bx, by, side, c.bw, ch, predicted,
        [&](std::uint32_t sym, std::uint32_t nbits, std::uint32_t raw, std::uint32_t context) {
            const std::uint8_t cl{runtime_ac_cluster(c.context_map, context)};
            const std::size_t idx{static_cast<std::size_t>(cl) * AC_HISTOGRAM_SIZE + sym};
            w.put(c.depth[idx], c.bits[idx]);
            if (nbits) {
                w.put(nbits, raw);
            }
        });
}

__global__ void group_size_kernel(const std::int16_t* ac, const std::int8_t* acs,
                                  const std::int32_t* nz_grid, std::size_t bw, std::size_t bh,
                                  std::size_t nblocks, std::size_t xg, std::size_t num_groups,
                                  const std::uint8_t* depth, const std::uint8_t* context_map,
                                  std::uint32_t* group_sizes, unsigned long long* run_offsets) {
    const std::size_t g{blockIdx.x};
    if (g >= num_groups) {
        return;
    }
    const AcGroupCtx c{
        build_ac_ctx(g, ac, acs, nz_grid, bw, bh, nblocks, xg, depth, nullptr, context_map)};
    const std::size_t r{(c.n_items + blockDim.x - 1) / blockDim.x};
    const std::size_t k0{min(static_cast<std::size_t>(threadIdx.x) * r, c.n_items)};
    const std::size_t k1{min(k0 + r, c.n_items)};
    unsigned long long run_bits{0};
    for (std::size_t k{k0}; k < k1; ++k) {
        run_bits += ac_item_bits(c, k);
    }
    __shared__ unsigned long long sh[AC_EMIT_THREADS];
    unsigned long long total{0};
    const unsigned long long run_off{block_scan_exclusive(run_bits, sh, total)};
    run_offsets[g * blockDim.x + threadIdx.x] = run_off;
    if (threadIdx.x == 0) {
        group_sizes[g] = static_cast<std::uint32_t>((total + 7) / 8);
    }
}

__global__ void group_emit_kernel(const std::int16_t* ac, const std::int8_t* acs,
                                  const std::int32_t* nz_grid, std::size_t bw, std::size_t bh,
                                  std::size_t nblocks, std::size_t xg, std::size_t num_groups,
                                  const std::uint8_t* depth, const std::uint16_t* bits_table,
                                  const std::uint8_t* context_map,
                                  const unsigned long long* run_offsets, std::uint8_t* out,
                                  const std::uint32_t* group_offsets) {
    const std::size_t g{blockIdx.x};
    if (g >= num_groups) {
        return;
    }
    const AcGroupCtx c{
        build_ac_ctx(g, ac, acs, nz_grid, bw, bh, nblocks, xg, depth, bits_table, context_map)};
    const std::size_t r{(c.n_items + blockDim.x - 1) / blockDim.x};
    const std::size_t k0{min(static_cast<std::size_t>(threadIdx.x) * r, c.n_items)};
    const std::size_t k1{min(k0 + r, c.n_items)};
    const unsigned long long run_off{run_offsets[g * blockDim.x + threadIdx.x]};
    AtomicBitWriter w{reinterpret_cast<unsigned int*>(out),
                      (static_cast<unsigned long long>(group_offsets[g]) << 3) + run_off};
    for (std::size_t k{k0}; k < k1; ++k) {
        ac_item_emit(c, k, w);
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

bool ac_build_histogram(const std::int16_t* ac, const std::int8_t* acs, std::size_t width,
                        std::size_t height, std::uint32_t* histogram) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t nblocks{bw * bh};
    if (cudaMemset(histogram, 0, AC_CLUSTER_HIST * sizeof(std::uint32_t)) != cudaSuccess) {
        return false;
    }
    std::int32_t* nz_grid{nullptr};
    if (cudaMalloc(&nz_grid, 3 * nblocks * sizeof(std::int32_t)) != cudaSuccess) {
        return false;
    }
    const unsigned int gthreads{256};
    ac_nzeros_grid_kernel<<<static_cast<unsigned int>((bw * bh + gthreads - 1) / gthreads),
                            gthreads>>>(ac, acs, bw, bh, nblocks, nz_grid);
    const std::size_t total{3 * bw * bh};
    const unsigned int threads{256};
    const unsigned int blocks{static_cast<unsigned int>((total + threads - 1) / threads)};
    histogram_kernel<<<blocks, threads>>>(ac, acs, nz_grid, bw, bh, nblocks, histogram);
    const bool ok{cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess};
    cudaFree(nz_grid);
    return ok;
}

static bool ac_encode_groups_impl(const std::int16_t* ac, const std::int8_t* acs, std::size_t width,
                                  std::size_t height, const std::uint8_t* context_map,
                                  const std::uint8_t* depth, const std::uint16_t* bits,
                                  std::size_t num_clusters, std::uint8_t* out,
                                  std::size_t out_capacity, std::uint32_t* group_sizes,
                                  std::uint32_t* group_offsets, std::size_t* total_bytes) {
    ensure_constants();
    if (num_clusters == 0 || num_clusters > 256) {
        return false;
    }
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t nblocks{bw * bh};
    const std::size_t xg{(bw + AC_GROUP_BLOCKS - 1) / AC_GROUP_BLOCKS};
    const std::size_t num_groups{ac_num_groups(width, height)};

    unsigned long long* run_offsets{nullptr};
    std::int32_t* nz_grid{nullptr};
    if (cudaMallocAsync(&run_offsets, num_groups * AC_EMIT_THREADS * sizeof(unsigned long long),
                        0) != cudaSuccess ||
        cudaMallocAsync(&nz_grid, 3 * nblocks * sizeof(std::int32_t), 0) != cudaSuccess) {
        cudaFreeAsync(run_offsets, 0);
        cudaFreeAsync(nz_grid, 0);
        return false;
    }
    const auto free_run = [&] {
        cudaFreeAsync(run_offsets, 0);
        cudaFreeAsync(nz_grid, 0);
    };

    const unsigned int gthreads{256};
    ac_nzeros_grid_kernel<<<static_cast<unsigned int>((bw * bh + gthreads - 1) / gthreads),
                            gthreads>>>(ac, acs, bw, bh, nblocks, nz_grid);

    const unsigned int grid{static_cast<unsigned int>(num_groups)};
    group_size_kernel<<<grid, AC_EMIT_THREADS>>>(ac, acs, nz_grid, bw, bh, nblocks, xg, num_groups,
                                                 depth, context_map, group_sizes, run_offsets);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        free_run();
        return false;
    }

    void* d_temp{nullptr};
    std::size_t temp_bytes{0};
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, group_sizes, group_offsets,
                                  static_cast<int>(num_groups));
    if (cudaMallocAsync(&d_temp, temp_bytes, 0) != cudaSuccess) {
        free_run();
        return false;
    }
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, group_sizes, group_offsets,
                                  static_cast<int>(num_groups));
    const cudaError_t scan_status{cudaDeviceSynchronize()};
    cudaFreeAsync(d_temp, 0);
    if (scan_status != cudaSuccess) {
        free_run();
        return false;
    }

    std::uint32_t last_offset{0};
    std::uint32_t last_size{0};
    if (cudaMemcpy(&last_offset, group_offsets + (num_groups - 1), sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&last_size, group_sizes + (num_groups - 1), sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        free_run();
        return false;
    }
    *total_bytes = static_cast<std::size_t>(last_offset) + last_size;
    if (*total_bytes > out_capacity) {
        free_run();
        return false;
    }

    const std::size_t zero_bytes{(*total_bytes + 3) & ~std::size_t{3}};
    if (cudaMemset(out, 0, zero_bytes) != cudaSuccess) {
        free_run();
        return false;
    }

    group_emit_kernel<<<grid, AC_EMIT_THREADS>>>(ac, acs, nz_grid, bw, bh, nblocks, xg, num_groups,
                                                 depth, bits, context_map, run_offsets, out,
                                                 group_offsets);
    const bool ok{cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess};
    free_run();
    return ok;
}

bool ac_encode_groups(const std::int16_t* ac, const std::int8_t* acs, std::size_t width,
                      std::size_t height, const std::uint8_t* depth, const std::uint16_t* bits,
                      std::uint8_t* out, std::size_t out_capacity, std::uint32_t* group_sizes,
                      std::uint32_t* group_offsets, std::size_t* total_bytes) {
    return ac_encode_groups_impl(ac, acs, width, height, nullptr, depth, bits, AC_NUM_CLUSTERS, out,
                                 out_capacity, group_sizes, group_offsets, total_bytes);
}

std::size_t dc_num_groups(std::size_t width, std::size_t height) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t ydg{(bh + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    return xdg * ydg;
}

bool dc_build_histograms(const std::int32_t* dc, std::size_t width, std::size_t height,
                         std::uint32_t* histograms) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t nblocks{bw * bh};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t num_groups{dc_num_groups(width, height)};
    if (cudaMemset(histograms, 0, num_groups * AC_HISTOGRAM_SIZE * sizeof(std::uint32_t)) !=
        cudaSuccess) {
        return false;
    }
    const std::size_t total{3 * bw * bh};
    const unsigned int threads{256};
    const unsigned int blocks{static_cast<unsigned int>((total + threads - 1) / threads)};
    dc_histogram_kernel<<<blocks, threads>>>(dc, bw, bh, nblocks, xdg, histograms);
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

bool acmeta_build_histograms(const std::int32_t* quant_field, const std::int8_t* acs,
                             const std::int8_t* ytox_map, const std::int8_t* ytob_map,
                             std::size_t width, std::size_t height, std::uint32_t* histograms) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t cmw{(bw + 7) / 8};
    const std::size_t cmh{(bh + 7) / 8};
    const std::size_t num_groups{dc_num_groups(width, height)};
    if (cudaMemset(histograms, 0, num_groups * AC_HISTOGRAM_SIZE * sizeof(std::uint32_t)) !=
        cudaSuccess) {
        return false;
    }
    const unsigned int threads{256};
    const unsigned int blocks{static_cast<unsigned int>((bw * bh + threads - 1) / threads)};
    acmeta_acsqf_histogram_kernel<<<blocks, threads>>>(quant_field, acs, bw, bh, xdg, histograms);
    const unsigned int cfl_blocks{static_cast<unsigned int>((cmw * cmh + threads - 1) / threads)};
    acmeta_cfl_histogram_kernel<<<cfl_blocks, threads>>>(ytox_map, ytob_map, cmw, cmh, xdg,
                                                         histograms);
    const unsigned int zero_blocks{static_cast<unsigned int>((num_groups + threads - 1) / threads)};
    acmeta_epf_zeros_kernel<<<zero_blocks, threads>>>(bw, bh, xdg, num_groups, histograms);
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

bool dc_encode_groups(const std::int32_t* dc, std::size_t width, std::size_t height,
                      const std::int32_t* quant_field, const std::int8_t* acs,
                      const std::int8_t* ytox_map, const std::int8_t* ytob_map,
                      const std::uint8_t* dc_depth, const std::uint16_t* dc_bits,
                      const std::uint8_t* acmeta_depth, const std::uint16_t* acmeta_bits,
                      const std::uint8_t* blob_pre, const std::uint32_t* blob_pre_off,
                      const std::uint32_t* blob_pre_bits, const std::uint8_t* blob_mid,
                      const std::uint32_t* blob_mid_off, const std::uint32_t* blob_mid_bits,
                      std::uint8_t* out, std::size_t out_capacity, std::uint32_t* group_sizes,
                      std::uint32_t* group_offsets, std::size_t* total_bytes) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t nblocks{bw * bh};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t cmw{(bw + 7) / 8};
    const std::size_t num_groups{dc_num_groups(width, height)};

    // Per-group first-block position lists for the AcMetadata ACS/QF columns.
    const std::size_t fb_stride{DC_GROUP_BLOCKS * DC_GROUP_BLOCKS};
    std::size_t* fb_pos{nullptr};
    std::size_t* fb_count{nullptr};
    unsigned long long* chunk_bits{nullptr};
    unsigned long long* chunk_base{nullptr};
    if (cudaMallocAsync(&fb_pos, num_groups * fb_stride * sizeof(std::size_t), 0) != cudaSuccess ||
        cudaMallocAsync(&fb_count, num_groups * sizeof(std::size_t), 0) != cudaSuccess ||
        cudaMallocAsync(&chunk_bits, num_groups * DC_CHUNKS * sizeof(unsigned long long), 0) !=
            cudaSuccess ||
        cudaMallocAsync(&chunk_base, num_groups * DC_CHUNKS * sizeof(unsigned long long), 0) !=
            cudaSuccess) {
        return false;
    }
    const auto free_chunks = [&] {
        cudaFreeAsync(fb_pos, 0);
        cudaFreeAsync(fb_count, 0);
        cudaFreeAsync(chunk_bits, 0);
        cudaFreeAsync(chunk_base, 0);
    };

    const unsigned int fb_threads{64};
    const unsigned int fb_blocks{
        static_cast<unsigned int>((num_groups + fb_threads - 1) / fb_threads)};
    build_first_blocks_kernel<<<fb_blocks, fb_threads>>>(acs, bw, bh, xdg, num_groups, fb_stride,
                                                         fb_pos, fb_count);

    const dim3 chunk_grid{static_cast<unsigned int>(num_groups), DC_CHUNKS};
    dc_chunk_size_kernel<<<chunk_grid, DC_EMIT_THREADS>>>(
        dc, bw, bh, nblocks, xdg, num_groups, quant_field, acs, ytox_map, ytob_map, cmw, fb_pos,
        fb_count, fb_stride, dc_depth, dc_bits, acmeta_depth, acmeta_bits, blob_pre, blob_pre_off,
        blob_pre_bits, blob_mid, blob_mid_off, blob_mid_bits, chunk_bits);
    dc_group_scan_kernel<<<static_cast<unsigned int>(num_groups), DC_CHUNKS>>>(
        num_groups, chunk_bits, chunk_base, group_sizes);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        free_chunks();
        return false;
    }

    void* d_temp{nullptr};
    std::size_t temp_bytes{0};
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, group_sizes, group_offsets,
                                  static_cast<int>(num_groups));
    if (cudaMallocAsync(&d_temp, temp_bytes, 0) != cudaSuccess) {
        free_chunks();
        return false;
    }
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, group_sizes, group_offsets,
                                  static_cast<int>(num_groups));
    const cudaError_t scan_status{cudaDeviceSynchronize()};
    cudaFreeAsync(d_temp, 0);
    if (scan_status != cudaSuccess) {
        free_chunks();
        return false;
    }

    std::uint32_t last_offset{0};
    std::uint32_t last_size{0};
    if (cudaMemcpy(&last_offset, group_offsets + (num_groups - 1), sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&last_size, group_sizes + (num_groups - 1), sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        free_chunks();
        return false;
    }
    *total_bytes = static_cast<std::size_t>(last_offset) + last_size;
    if (*total_bytes > out_capacity) {
        free_chunks();
        return false;
    }

    // The emit kernel composes bits with atomicOr (runs sharing a byte), so the
    // section region must start zeroed. Round up to the 32-bit word the last bits
    // land in; the wrapper's capacity has slack for the padding.
    const std::size_t zero_bytes{(*total_bytes + 3) & ~std::size_t{3}};
    if (cudaMemset(out, 0, zero_bytes) != cudaSuccess) {
        free_chunks();
        return false;
    }

    dc_chunk_emit_kernel<<<chunk_grid, DC_EMIT_THREADS>>>(
        dc, bw, bh, nblocks, xdg, num_groups, quant_field, acs, ytox_map, ytob_map, cmw, fb_pos,
        fb_count, fb_stride, dc_depth, dc_bits, acmeta_depth, acmeta_bits, blob_pre, blob_pre_off,
        blob_pre_bits, blob_mid, blob_mid_off, blob_mid_bits, chunk_base, out, group_offsets);
    const bool ok{cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess};
    free_chunks();
    return ok;
}

}  // namespace cujpegxl
