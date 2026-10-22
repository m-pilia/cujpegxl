// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy.h"

#include <mutex>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "ac_context.h"
#include "coeff_order.h"
#include "dc_predict.h"
#include "src/bitstream/token_stream.h"
#include "vardct_layout.h"
#include "warp_scan.cuh"

namespace cujpegxl {
namespace {

// libjxl default HybridUintConfig (4, 2, 0), matching the host bitstream writer.
constexpr std::uint32_t SPLIT_EXPONENT = 4;
constexpr std::uint32_t MSB_IN_TOKEN = 2;
constexpr std::uint32_t LSB_IN_TOKEN = 0;

// Physical AC channel order Y, X, B (decoder LoadBlock order) indexing q's
// planes 0=X, 1=Y, 2=B.
__constant__ int CHANNEL_ORDER[3]{1, 0, 2};

// order[k] = libjxl-raster index of the k-th coefficient in the square DCT scan
// order, per transform size. NATURAL_ORDER is the 8x8 table; the DCT16/DCT32
// tables drive the larger blocks of the mixed-block tokenizer. Global rather
// than __constant__: the warp tokenizer indexes them with a different k per
// lane, which the constant cache would serialize.
__device__ std::uint32_t NATURAL_ORDER[64];
__device__ std::uint32_t NATURAL_ORDER_16[256];
__device__ std::uint32_t NATURAL_ORDER_32[1024];

__device__ std::int32_t AC_LOG2_TABLE[AC_LOG2_TABLE_SIZE + 1];

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
    const AcLog2Table log2_table{ac_log2_table()};
    cudaMemcpyToSymbol(AC_LOG2_TABLE, log2_table.data(), sizeof(log2_table));
}

// The copies run on the legacy default stream; the wait orders them before the
// kernels of the non-blocking streams.
void ensure_constants() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        init_constants();
        cudaStreamSynchronize(nullptr);
    });
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
    const std::uint32_t n{31u - static_cast<std::uint32_t>(__clz(value))};
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

// The first-block lists are compacted one block row at a time: a thread block
// per (block row, DcGroup column) with one thread per row position. A first
// pass stores each row's count. The second pass places a position at the counts
// of its group's earlier rows, plus the counts of the earlier warps in its row,
// plus its ballot rank, which reproduces the group raster order.
constexpr unsigned int FIRST_BLOCK_THREADS = static_cast<unsigned int>(DC_GROUP_BLOCKS);
constexpr unsigned int FIRST_BLOCK_WARPS = FIRST_BLOCK_THREADS / 32;

struct FirstBlockSlot {
    std::size_t g;
    std::size_t row;
    std::size_t block;
    bool flag;
};

__device__ FirstBlockSlot first_block_slot(const std::int8_t* acs, std::size_t bw,
                                           std::size_t xdg) {
    const std::size_t by{blockIdx.x};
    const std::size_t gx{blockIdx.y};
    const std::size_t bx{gx * DC_GROUP_BLOCKS + threadIdx.x};
    FirstBlockSlot s{};
    s.g = (by / DC_GROUP_BLOCKS) * xdg + gx;
    s.row = by % DC_GROUP_BLOCKS;
    s.block = by * bw + bx;
    s.flag = bx < bw && (acs == nullptr || acs[s.block] != ACS_COVERED);
    return s;
}

__global__ void first_block_row_count_kernel(const std::int8_t* acs, std::size_t bw,
                                             std::size_t xdg, std::uint32_t* row_counts) {
    const FirstBlockSlot s{first_block_slot(acs, bw, xdg)};
    const int count{__syncthreads_count(s.flag)};
    if (threadIdx.x == 0) {
        row_counts[s.g * DC_GROUP_BLOCKS + s.row] = static_cast<std::uint32_t>(count);
    }
}

__global__ void first_block_scatter_kernel(const std::int8_t* acs, std::size_t bw, std::size_t bh,
                                           std::size_t xdg, const std::uint32_t* row_counts,
                                           std::size_t* positions, std::size_t* counts) {
    const FirstBlockSlot s{first_block_slot(acs, bw, xdg)};
    const unsigned int lane{threadIdx.x % 32};
    const unsigned int warp{threadIdx.x / 32};
    const std::uint32_t earlier_row{
        threadIdx.x < s.row ? row_counts[s.g * DC_GROUP_BLOCKS + threadIdx.x] : 0u};
    const std::uint32_t earlier_rows{warp_inclusive_sum(earlier_row, lane)};
    const unsigned int flags{__ballot_sync(~0u, s.flag)};
    __shared__ std::uint32_t warp_earlier_rows[FIRST_BLOCK_WARPS];
    __shared__ std::uint32_t warp_flags[FIRST_BLOCK_WARPS];
    if (lane == 31) {
        warp_earlier_rows[warp] = earlier_rows;
        warp_flags[warp] = static_cast<std::uint32_t>(__popc(flags));
    }
    __syncthreads();
    std::uint32_t row_base{0};
    std::uint32_t warp_base{0};
    std::uint32_t row_count{0};
    for (unsigned int w{0}; w < FIRST_BLOCK_WARPS; ++w) {
        row_base += warp_earlier_rows[w];
        warp_base += w < warp ? warp_flags[w] : 0u;
        row_count += warp_flags[w];
    }
    if (s.flag) {
        const std::uint32_t rank{static_cast<std::uint32_t>(__popc(flags & ((1u << lane) - 1u)))};
        positions[s.g * DC_FIRST_BLOCK_STRIDE + row_base + warp_base + rank] = s.block;
    }
    const std::size_t by{blockIdx.x};
    const bool last_row{by + 1 == bh || s.row + 1 == DC_GROUP_BLOCKS};
    if (threadIdx.x == 0 && last_row) {
        counts[s.g] = static_cast<std::size_t>(row_base) + row_count;
    }
}

constexpr int DC_EMIT_THREADS = 512;

// A DcGroup holds up to 256x256 blocks, so a single thread block per group leaves
// most of the device idle (a 4K frame has only 4 DcGroups). Each group is instead
// split into DC_CHUNKS contiguous item ranges emitted by separate thread blocks;
// a per-group scan of the chunk bit counts gives each chunk its bit offset within
// the group so the concatenation still reproduces the reference bitstream. The
// size pass also stores each thread's run offset within its chunk, so the emit
// pass tokenizes every item only once.
constexpr int DC_CHUNKS = 128;

// A DcGroup is emitted cooperatively by one thread block: its section is an
// ordered sequence of `m` items (blob_pre, then the DC tokens, then blob_mid,
// then the AcMetadata tokens), each contributing a known number of bits. Threads
// own contiguous runs of items, a block-wide scan of per-run bit counts gives
// each run's bit offset, and each item is written at its offset via atomicOr so
// runs that share a byte compose without a race. DcCtx caches the per-group
// geometry and the AcMetadata channel-sample structure: YtoX (cw*ch), YtoB
// (cw*ch), ACS+QF row 0 (count strategy samples), row 1 (count quant-field
// samples), matching the host reference. The EPF channel that follows has a
// single-symbol leaf of its own and codes no bits, so it has no items.
// `acs`/`ytox_map`/
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
    std::size_t n;              // DC samples per channel = gbw*gbh
    std::size_t cw, ch;         // CfL color-tile grid within the group
    std::size_t count;          // first-blocks (ACS+QF columns)
    std::size_t m;              // total items
};

// (symbol, extra bits) of the group's `j`-th AcMetadata sample, in channel/row
// order: YtoX, YtoB, ACS row 0, QF row 1. Predictor::Zero, so each sample is
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
    } else {
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
                              const std::uint8_t* dc_depth, const std::uint16_t* dc_bits,
                              const std::uint8_t* acmeta_depth, const std::uint16_t* acmeta_bits,
                              const std::uint8_t* blob_pre, const std::uint32_t* blob_pre_off,
                              const std::uint32_t* blob_pre_bits, const std::uint8_t* blob_mid,
                              const std::uint32_t* blob_mid_off,
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
    c.fb_pos = fb_pos + g * DC_FIRST_BLOCK_STRIDE;
    c.n = e.gbw * e.gbh;
    c.cw = (e.gbw + 7) / 8;
    c.ch = (e.gbh + 7) / 8;
    c.count = fb_count[g];
    c.m = 3 * c.n + 2 + 2 * c.cw * c.ch + 2 * c.count;
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

// LSB-first bit packer for a thread's single contiguous run of bits. A run owns a
// known absolute bit range [pos, end); its interior 32-bit words belong to it
// alone and are written with plain stores, while the (at most one) word at each
// end that a run shares with an adjacent run is composed with atomicOr. Since the
// output region is pre-zeroed and the shared bit sub-ranges are disjoint, the OR
// reproduces the reference byte layout with two atomics per run instead of one
// per symbol chunk. Bits are staged in `acc` (valid low bits counted by `fill`);
// each `put` appends up to 32 bits and flushes any completed words. `finish` must
// be called once after the run's items to emit the trailing partial word.
struct RunBitWriter {
    unsigned int* words;
    unsigned long long word_idx;
    unsigned long long acc;
    std::uint32_t fill;
    bool first_shared;

    __device__ RunBitWriter(unsigned int* out_words, unsigned long long pos)
        : words{out_words},
          word_idx{pos >> 5},
          acc{0},
          fill{static_cast<std::uint32_t>(pos & 31)},
          first_shared{(pos & 31) != 0} {}

    __device__ void put(std::uint32_t n_bits, std::uint32_t value) {
        const std::uint32_t mask{n_bits >= 32 ? ~0u : ((1u << n_bits) - 1u)};
        acc |= static_cast<unsigned long long>(value & mask) << fill;
        fill += n_bits;
        while (fill >= 32) {
            const unsigned int word{static_cast<unsigned int>(acc)};
            if (first_shared) {
                if (word != 0) {
                    atomicOr(&words[word_idx], word);
                }
                first_shared = false;
            } else {
                words[word_idx] = word;
            }
            acc >>= 32;
            fill -= 32;
            ++word_idx;
        }
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

    // The trailing partial word (if any) is shared with the next run, so it is
    // always composed via atomicOr rather than stored.
    __device__ void finish() {
        if (fill != 0) {
            const unsigned int word{static_cast<unsigned int>(acc)};
            if (word != 0) {
                atomicOr(&words[word_idx], word);
            }
        }
    }
};

// Emits item k at the writer's current position.
__device__ void dc_item_emit(const DcCtx& c, std::size_t k, RunBitWriter& w) {
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

// End of a section of byte-aligned groups laid out at `group_offsets`, read on
// the device so that the host never waits for the section total.
__device__ std::size_t section_end(const std::uint32_t* group_offsets,
                                   const std::uint32_t* group_sizes, std::size_t num_groups) {
    return static_cast<std::size_t>(group_offsets[num_groups - 1]) + group_sizes[num_groups - 1];
}

// The emit kernels compose bits with atomicOr (runs sharing a byte), so the
// section must start zeroed, up to the 32-bit word the last bits land in (the
// wrappers' capacities have slack for the padding). A section over `capacity`
// is neither zeroed nor emitted; the caller detects it from the group sizes.
constexpr unsigned int ZERO_FILL_THREADS = 256;
constexpr unsigned int ZERO_FILL_BLOCKS = 256;

__global__ void zero_section_kernel(unsigned int* words, const std::uint32_t* group_offsets,
                                    const std::uint32_t* group_sizes, std::size_t num_groups,
                                    std::size_t capacity) {
    const std::size_t end{section_end(group_offsets, group_sizes, num_groups)};
    if (end > capacity) {
        return;
    }
    const std::size_t num_words{(end + 3) / 4};
    const std::size_t stride{static_cast<std::size_t>(gridDim.x) * blockDim.x};
    for (std::size_t i{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
         i < num_words; i += stride) {
        words[i] = 0;
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

// The DC and AcMetadata histograms are launched over (strip, DcGroup column),
// with a strip of a few block rows and one thread per row position, so every
// thread block counts into a single group's shared histogram and flushes it with
// one global atomic per non-zero bin. An AcMetadata strip is one 64x64
// color-tile row, so the CfL tiles of a strip are those of its thread block.
constexpr unsigned int GROUP_HIST_THREADS = static_cast<unsigned int>(DC_GROUP_BLOCKS);
constexpr std::size_t DC_HIST_ROWS = 4;
constexpr std::size_t ACMETA_HIST_ROWS = 8;
constexpr std::size_t COLOR_TILES_PER_DC_GROUP = DC_GROUP_BLOCKS / ACMETA_HIST_ROWS;
static_assert(DC_GROUP_BLOCKS % DC_HIST_ROWS == 0 && DC_GROUP_BLOCKS % ACMETA_HIST_ROWS == 0);

struct GroupHistStrip {
    std::size_t g;
    std::size_t bx;
    std::size_t bx0;
    std::size_t by0;
    std::size_t row_lo;
    std::size_t row_hi;
};

__device__ GroupHistStrip group_hist_strip(std::size_t rows, std::size_t bw, std::size_t bh,
                                           std::size_t xdg) {
    GroupHistStrip s{};
    s.row_lo = blockIdx.x * rows;
    s.row_hi = min(s.row_lo + rows, bh);
    s.by0 = (s.row_lo / DC_GROUP_BLOCKS) * DC_GROUP_BLOCKS;
    s.bx0 = blockIdx.y * DC_GROUP_BLOCKS;
    s.bx = s.bx0 + threadIdx.x;
    s.g = (s.row_lo / DC_GROUP_BLOCKS) * xdg + blockIdx.y;
    return s;
}

__device__ void clear_group_histogram(unsigned int* sh) {
    for (unsigned int i{threadIdx.x}; i < AC_HISTOGRAM_SIZE; i += blockDim.x) {
        sh[i] = 0;
    }
    __syncthreads();
}

__device__ void flush_group_histogram(const unsigned int* sh, std::uint32_t* histogram) {
    __syncthreads();
    for (unsigned int i{threadIdx.x}; i < AC_HISTOGRAM_SIZE; i += blockDim.x) {
        if (sh[i] != 0) {
            atomicAdd(&histogram[i], sh[i]);
        }
    }
}

__device__ void count_signed_token(std::int32_t value, unsigned int* sh) {
    std::uint32_t symbol{}, nbits{}, bits{};
    hybrid_encode(pack_signed(value), symbol, nbits, bits);
    atomicAdd(&sh[symbol], 1u);
}

__global__ void dc_histogram_kernel(const std::int32_t* dc_buf, std::size_t bw, std::size_t bh,
                                    std::size_t nblocks, std::size_t xdg,
                                    std::uint32_t* histograms) {
    __shared__ unsigned int sh[AC_HISTOGRAM_SIZE];
    clear_group_histogram(sh);
    const GroupHistStrip s{group_hist_strip(DC_HIST_ROWS, bw, bh, xdg)};
    if (s.bx < bw) {
        for (std::size_t by{s.row_lo}; by < s.row_hi; ++by) {
            for (int c{0}; c < 3; ++c) {
                const std::int32_t dc{
                    dc_buf[static_cast<std::size_t>(c) * nblocks + by * bw + s.bx]};
                count_signed_token(
                    dc - dc_group_gradient(dc_buf, c, nblocks, bw, s.bx, by, s.bx0, s.by0), sh);
            }
        }
    }
    flush_group_histogram(sh, histograms + s.g * AC_HISTOGRAM_SIZE);
}

// Counts the strip's AcMetadata samples outside the EPF channel: each
// first-block's ACS strategy token (row 0) and quant-field token (row 1), and
// each color tile's YtoX and YtoB tokens (null maps are the base correlation,
// symbol 0).
__global__ void acmeta_histogram_kernel(const std::int32_t* quant_field, const std::int8_t* acs,
                                        const std::int8_t* ytox_map, const std::int8_t* ytob_map,
                                        std::size_t bw, std::size_t bh, std::size_t cmw,
                                        std::size_t xdg, std::uint32_t* histograms) {
    __shared__ unsigned int sh[AC_HISTOGRAM_SIZE];
    clear_group_histogram(sh);
    const GroupHistStrip s{group_hist_strip(ACMETA_HIST_ROWS, bw, bh, xdg)};
    if (s.bx < bw) {
        for (std::size_t by{s.row_lo}; by < s.row_hi; ++by) {
            const std::size_t block{by * bw + s.bx};
            const int side{acs == nullptr ? 8 : acs[block]};
            if (side != ACS_COVERED) {
                count_signed_token(raw_strategy_dev(side), sh);
                count_signed_token(quant_field[block] - 1, sh);
            }
        }
    }
    const std::size_t ctx{blockIdx.y * COLOR_TILES_PER_DC_GROUP + threadIdx.x};
    if (threadIdx.x < COLOR_TILES_PER_DC_GROUP && ctx < cmw) {
        const std::size_t tile{blockIdx.x * cmw + ctx};
        count_signed_token(ytox_map == nullptr ? 0 : ytox_map[tile], sh);
        count_signed_token(ytob_map == nullptr ? 0 : ytob_map[tile], sh);
    }
    flush_group_histogram(sh, histograms + s.g * AC_HISTOGRAM_SIZE);
}

dim3 group_hist_grid(std::size_t rows, std::size_t bh, std::size_t xdg) {
    return dim3{static_cast<unsigned int>((bh + rows - 1) / rows), static_cast<unsigned int>(xdg)};
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
    const std::size_t* fb_pos, const std::size_t* fb_count, const std::uint8_t* dc_depth,
    const std::uint16_t* dc_bits, const std::uint8_t* acmeta_depth,
    const std::uint16_t* acmeta_bits, const std::uint8_t* blob_pre,
    const std::uint32_t* blob_pre_off, const std::uint32_t* blob_pre_bits,
    const std::uint8_t* blob_mid, const std::uint32_t* blob_mid_off,
    const std::uint32_t* blob_mid_bits, std::uint32_t* run_offsets,
    unsigned long long* chunk_bits) {
    const std::size_t g{blockIdx.x};
    if (g >= num_groups) {
        return;
    }
    const DcCtx c{build_dc_ctx(g, dc, bw, bh, nblocks, xdg, quant_field, acs, ytox_map, ytob_map,
                               cmw, fb_pos, fb_count, dc_depth, dc_bits, acmeta_depth, acmeta_bits,
                               blob_pre, blob_pre_off, blob_pre_bits, blob_mid, blob_mid_off,
                               blob_mid_bits)};
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
    const std::size_t chunk{g * DC_CHUNKS + blockIdx.y};
    run_offsets[chunk * DC_EMIT_THREADS + threadIdx.x] = static_cast<std::uint32_t>(run_off);
    if (threadIdx.x == 0) {
        chunk_bits[chunk] = total;
    }
}

// One block walks the few DcGroups (4 at 4K) in order: the exclusive scan of a
// group's chunk bit counts gives each chunk's bit offset within the group, and
// the group byte sizes and offsets come out of the same launch instead of a
// separate device-wide scan.
__global__ void dc_chunk_scan_kernel(std::size_t num_groups, const unsigned long long* chunk_bits,
                                     unsigned long long* chunk_base, std::uint32_t* group_sizes,
                                     std::uint32_t* group_offsets) {
    __shared__ unsigned long long sh[DC_CHUNKS];
    std::uint32_t offset{0};
    for (std::size_t g{0}; g < num_groups; ++g) {
        const std::size_t chunk{g * DC_CHUNKS + threadIdx.x};
        unsigned long long group_bits{0};
        chunk_base[chunk] = block_scan_exclusive(chunk_bits[chunk], sh, group_bits);
        const auto size{static_cast<std::uint32_t>((group_bits + 7) / 8)};
        if (threadIdx.x == 0) {
            group_sizes[g] = size;
            group_offsets[g] = offset;
        }
        offset += size;
    }
}

__global__ void dc_chunk_emit_kernel(
    const std::int32_t* dc, std::size_t bw, std::size_t bh, std::size_t nblocks, std::size_t xdg,
    std::size_t num_groups, const std::int32_t* quant_field, const std::int8_t* acs,
    const std::int8_t* ytox_map, const std::int8_t* ytob_map, std::size_t cmw,
    const std::size_t* fb_pos, const std::size_t* fb_count, const std::uint8_t* dc_depth,
    const std::uint16_t* dc_bits, const std::uint8_t* acmeta_depth,
    const std::uint16_t* acmeta_bits, const std::uint8_t* blob_pre,
    const std::uint32_t* blob_pre_off, const std::uint32_t* blob_pre_bits,
    const std::uint8_t* blob_mid, const std::uint32_t* blob_mid_off,
    const std::uint32_t* blob_mid_bits, const std::uint32_t* run_offsets,
    const unsigned long long* chunk_base, std::uint8_t* out, std::size_t out_capacity,
    const std::uint32_t* group_sizes, const std::uint32_t* group_offsets) {
    const std::size_t g{blockIdx.x};
    if (g >= num_groups || section_end(group_offsets, group_sizes, num_groups) > out_capacity) {
        return;
    }
    const DcCtx c{build_dc_ctx(g, dc, bw, bh, nblocks, xdg, quant_field, acs, ytox_map, ytob_map,
                               cmw, fb_pos, fb_count, dc_depth, dc_bits, acmeta_depth, acmeta_bits,
                               blob_pre, blob_pre_off, blob_pre_bits, blob_mid, blob_mid_off,
                               blob_mid_bits)};
    std::size_t k_lo{0};
    std::size_t k_hi{0};
    dc_chunk_range(c, blockIdx.y, k_lo, k_hi);
    std::size_t a0{0};
    std::size_t a1{0};
    dc_thread_range(k_lo, k_hi, a0, a1);
    const std::size_t chunk{g * DC_CHUNKS + blockIdx.y};
    const unsigned long long start{(static_cast<unsigned long long>(group_offsets[g]) << 3) +
                                   chunk_base[chunk] +
                                   run_offsets[chunk * DC_EMIT_THREADS + threadIdx.x]};
    RunBitWriter w{reinterpret_cast<unsigned int*>(out), start};
    for (std::size_t k{a0}; k < a1; ++k) {
        dc_item_emit(c, k, w);
    }
    w.finish();
}

// --- Mixed-block AC path ------------------------------------------------------
// The coefficient buffer is the covered-block layout (COEFFS_PER_BLOCK slots
// per 8x8 position, per channel plane nblocks*COEFFS_PER_BLOCK); a first-block of
// side N tokenizes order[covered..size) of its channel plane, gathered via
// covered_plane_slot; covered blocks emit nothing.

// Symbol and context of a block/channel's non-zero-count token, its first token.
// `nzeros` is the exact non-zero count of order[covered..size), precomputed by
// quantize_residual so the count-scan is not repeated in every pass.
__device__ void ac_count_token(int side, int c, std::uint32_t predicted, std::uint32_t nzeros,
                               std::uint32_t& symbol, std::uint32_t& nbits, std::uint32_t& bits,
                               std::uint32_t& context) {
    hybrid_encode(nzeros, symbol, nbits, bits);
    context = ac_nonzero_context(predicted, ac_block_context(c, ac_strategy_order(side)));
}

// Warp-cooperative coefficient tokens of one block/channel (the tokens after the
// non-zero count, in decoder order): lane i takes scan index k0 + i of
// each 32-wide chunk. A token's context depends on the non-zeros left before it
// and on whether its predecessor is non-zero, both prefixes over the scan order,
// so they come from a ballot of the chunk's non-zero flags instead of a serial
// walk. Every lane calls emit(active, symbol, nbits, bits, context) in lockstep
// (so `emit` may use warp collectives); `active` marks the lanes holding a token,
// in scan order by lane. The next chunk's coefficients are loaded before the
// current chunk is emitted, so their latency overlaps the emit.
template <typename Emit>
__device__ void warp_for_each_coeff_token(const std::int16_t* plane, std::size_t bx, std::size_t by,
                                          int side, std::size_t bw, int c, std::uint32_t nzeros,
                                          unsigned int lane, Emit emit) {
    const int cx{side / 8};
    const std::uint32_t covered{static_cast<std::uint32_t>(cx * cx)};
    const std::uint32_t size{static_cast<std::uint32_t>(side * side)};
    const std::uint32_t l2{side == 16 ? 2u : (side == 32 ? 4u : 0u)};
    const std::uint32_t* order{natural_order_dev(side)};
    const std::uint32_t histo_off{
        ac_zero_density_offset(ac_block_context(c, ac_strategy_order(side)))};
    const unsigned int lanes_below{(1u << lane) - 1u};
    const auto load = [&](std::uint32_t k) -> std::int32_t {
        return k < size ? plane[covered_plane_slot(side, bx, by, bw, __ldg(&order[k]))] : 0;
    };
    std::uint32_t remaining_before{nzeros};
    std::uint32_t prev_before{nzeros > size / 16 ? 0u : 1u};
    std::int32_t v{load(covered + lane)};
    for (std::uint32_t k0{covered}; k0 < size && remaining_before > 0; k0 += 32) {
        const std::uint32_t k{k0 + lane};
        const bool in_block{k < size};
        const unsigned int nz_mask{__ballot_sync(~0u, v != 0)};
        const std::uint32_t remaining{remaining_before -
                                      static_cast<std::uint32_t>(__popc(nz_mask & lanes_below))};
        const std::uint32_t prev{lane == 0 ? prev_before : (nz_mask >> (lane - 1)) & 1u};
        const bool active{in_block && remaining > 0};
        remaining_before -= static_cast<std::uint32_t>(__popc(nz_mask));
        prev_before = nz_mask >> 31;
        const std::int32_t v_next{remaining_before > 0 ? load(k + 32) : 0};
        std::uint32_t symbol{0}, nbits{0}, bits{0}, ctx{0};
        if (active) {
            ctx = histo_off + ac_zero_density_context(remaining, k, covered, l2, prev);
            hybrid_encode(pack_signed(v), symbol, nbits, bits);
        }
        emit(active, symbol, nbits, bits, ctx);
        v = v_next;
    }
}

__device__ std::uint32_t warp_sum(std::uint32_t v) {
    for (unsigned int d{16}; d > 0; d >>= 1) {
        v += __shfl_xor_sync(~0u, v, d);
    }
    return v;
}

// A token count and a raw bit (or word) count, packed so that one 64-bit sum
// adds both: the token count of a group or frame stays below 2^32.
__device__ unsigned long long pack_sizes(std::uint32_t tokens, std::uint32_t raw) {
    return (static_cast<unsigned long long>(raw) << 32) | tokens;
}

// Each warp takes 32 consecutive block/channel items: every lane counts its own
// item's non-zero-count token, then the warp tokenizes the coefficients of the
// items with non-zeros cooperatively, one item at a time. Each thread block
// covers HIST_ITEMS_PER_CTA items. The context histograms are too large for
// shared memory, so the tokens are counted with global atomics. The kernel also
// records every item's size (pack_sizes of its tokens and raw bits) at its index
// block * 3 + slot.
constexpr unsigned int HIST_THREADS = 512;
constexpr unsigned int HIST_ITEMS_PER_CTA = 1024;

__global__ void histogram_kernel(const std::int16_t* ac, const std::int8_t* acs,
                                 const std::int32_t* nz_grid, const std::int32_t* nz_exact,
                                 std::size_t bw, std::size_t bh, std::size_t nblocks,
                                 std::uint32_t* context_histograms,
                                 unsigned long long* item_sizes) {
    const unsigned int lane{threadIdx.x % 32};
    const std::size_t total{3 * bw * bh};
    const std::size_t first{static_cast<std::size_t>(blockIdx.x) * HIST_ITEMS_PER_CTA};
    const std::size_t last{min(first + HIST_ITEMS_PER_CTA, total)};
    const auto count = [&](bool active, std::uint32_t symbol, std::uint32_t, std::uint32_t,
                           std::uint32_t context) {
        if (active) {
            atomicAdd(&context_histograms[context * AC_CONTEXT_ALPHABET + symbol], 1u);
        }
    };
    for (std::size_t base{first + (threadIdx.x / 32) * 32}; base < last; base += blockDim.x) {
        const std::size_t idx{base + lane};
        const std::size_t block{idx / 3};
        const int side{idx < last ? (acs == nullptr ? 8 : acs[block]) : ACS_COVERED};
        const int c{CHANNEL_ORDER[idx % 3]};
        std::uint32_t nzeros{0};
        std::uint32_t tokens{0};
        std::uint32_t raw_bits{0};
        if (side != ACS_COVERED) {
            const std::size_t bx{block % bw};
            const std::size_t by{block / bw};
            nzeros =
                static_cast<std::uint32_t>(nz_exact[static_cast<std::size_t>(c) * nblocks + block]);
            std::uint32_t symbol{}, nbits{}, bits{}, context{};
            ac_count_token(side, c, ac_predicted_nzeros_dev(nz_grid, c, nblocks, bw, bx, by),
                           nzeros, symbol, nbits, bits, context);
            count(true, symbol, nbits, bits, context);
            tokens = 1;
            raw_bits = nbits;
        }
        unsigned int pending{__ballot_sync(~0u, side != ACS_COVERED && nzeros != 0)};
        while (pending != 0) {
            const int src{__ffs(pending) - 1};
            pending &= pending - 1;
            const std::size_t item_block{__shfl_sync(~0u, block, src)};
            const int item_side{__shfl_sync(~0u, side, src)};
            const int item_c{__shfl_sync(~0u, c, src)};
            const std::uint32_t item_nzeros{__shfl_sync(~0u, nzeros, src)};
            std::uint32_t item_tokens{0};
            std::uint32_t item_raw_bits{0};
            warp_for_each_coeff_token(
                ac + static_cast<std::size_t>(item_c) * nblocks * COEFFS_PER_BLOCK, item_block % bw,
                item_block / bw, item_side, bw, item_c, item_nzeros, lane,
                [&](bool active, std::uint32_t symbol, std::uint32_t nbits, std::uint32_t bits,
                    std::uint32_t context) {
                    count(active, symbol, nbits, bits, context);
                    item_tokens += static_cast<std::uint32_t>(__popc(__ballot_sync(~0u, active)));
                    item_raw_bits += active ? nbits : 0u;
                });
            item_raw_bits = warp_sum(item_raw_bits);
            if (lane == static_cast<unsigned int>(src)) {
                tokens += item_tokens;
                raw_bits += item_raw_bits;
            }
        }
        if (idx < last) {
            item_sizes[idx] = pack_sizes(tokens, raw_bits);
        }
    }
}

struct AcGroupCtx {
    const std::int16_t* ac;
    const std::int8_t* acs;
    const std::int32_t* nz_grid;
    const std::int32_t* nz_exact;
    std::size_t bw, nblocks;
    std::size_t bx0, by0, gbw;
    std::size_t n_items;  // gbw * gbh * 3, covered block-channels emit nothing
};

__device__ AcGroupCtx build_ac_ctx(std::size_t g, const std::int16_t* ac, const std::int8_t* acs,
                                   const std::int32_t* nz_grid, const std::int32_t* nz_exact,
                                   std::size_t bw, std::size_t bh, std::size_t nblocks,
                                   std::size_t xg) {
    const GroupExtent e{group_extent(g, bw, bh, xg)};
    AcGroupCtx c{};
    c.ac = ac;
    c.acs = acs;
    c.nz_grid = nz_grid;
    c.nz_exact = nz_exact;
    c.bw = bw;
    c.nblocks = nblocks;
    c.bx0 = e.bx0;
    c.by0 = e.by0;
    c.gbw = e.gbw;
    c.n_items = e.gbw * e.gbh * 3;
    return c;
}

// The (block, channel, predicted nzeros, exact nzeros) of item i, and its
// transform side (ACS_COVERED if the item contributes no tokens).
__device__ int ac_item(const AcGroupCtx& c, std::size_t i, std::size_t& bx, std::size_t& by,
                       int& ch, std::uint32_t& predicted, std::uint32_t& nzeros,
                       const std::int16_t*& plane) {
    const int p{static_cast<int>(i % 3)};
    const std::size_t bidx{i / 3};
    bx = c.bx0 + bidx % c.gbw;
    by = c.by0 + bidx / c.gbw;
    ch = CHANNEL_ORDER[p];
    const std::size_t block{by * c.bw + bx};
    const int side{c.acs == nullptr ? 8 : c.acs[block]};
    plane = c.ac + static_cast<std::size_t>(ch) * c.nblocks * COEFFS_PER_BLOCK;
    predicted = ac_predicted_nzeros_dev(c.nz_grid, ch, c.nblocks, c.bw, bx, by);
    nzeros =
        static_cast<std::uint32_t>(c.nz_exact[static_cast<std::size_t>(ch) * c.nblocks + block]);
    return side;
}

// A lane's item of a 32-item chunk, with its non-zero-count token; `side` is
// ACS_COVERED past the group's items or if the item emits nothing.
struct AcLaneItem {
    std::size_t bx, by;
    int ch, side;
    std::uint32_t nzeros;
    std::uint32_t symbol, nbits, bits, context;
};

__device__ AcLaneItem ac_lane_item(const AcGroupCtx& c, std::size_t i) {
    AcLaneItem it{};
    it.side = ACS_COVERED;
    if (i < c.n_items) {
        std::uint32_t predicted{};
        const std::int16_t* plane{};
        it.side = ac_item(c, i, it.bx, it.by, it.ch, predicted, it.nzeros, plane);
        if (it.side != ACS_COVERED) {
            ac_count_token(it.side, it.ch, predicted, it.nzeros, it.symbol, it.nbits, it.bits,
                           it.context);
        }
    }
    return it;
}

// Runs warp_for_each_coeff_token over lane `src`'s item.
template <typename Emit>
__device__ void warp_item_coeff_tokens(const AcGroupCtx& c, const AcLaneItem& it, int src,
                                       unsigned int lane, Emit emit) {
    const int ch{__shfl_sync(~0u, it.ch, src)};
    warp_for_each_coeff_token(c.ac + static_cast<std::size_t>(ch) * c.nblocks * COEFFS_PER_BLOCK,
                              __shfl_sync(~0u, it.bx, src), __shfl_sync(~0u, it.by, src),
                              __shfl_sync(~0u, it.side, src), c.bw, ch,
                              __shfl_sync(~0u, it.nzeros, src), lane, emit);
}

// One warp per 32-item chunk of an AC group (AC_GROUP_CHUNKS chunk slots per
// group; slots past a partial group's items are empty). The chunk sizes,
// scanned within each group and then over the groups, place every chunk's
// tokens and raw bits; the pack kernel scans the item sizes of its chunk to
// place every item.
constexpr unsigned int AC_CHUNK_THREADS = 128;
constexpr std::size_t AC_GROUP_ITEMS = AC_GROUP_BLOCKS * AC_GROUP_BLOCKS * 3;
constexpr int AC_GROUP_CHUNKS = static_cast<int>(AC_GROUP_ITEMS / 32);
constexpr unsigned int AC_LAYOUT_THREADS = 256;

// Raw bits of an AC token: a coefficient token's value is pack_signed of an
// int16 (at most 13 raw bits), a count token's at most 1008 (7 raw bits).
constexpr std::size_t AC_MAX_TOKEN_RAW_BITS = 13;

__global__ void ac_chunk_sizes_kernel(const unsigned long long* item_sizes, std::size_t bw,
                                      std::size_t bh, std::size_t xg, std::size_t num_groups,
                                      unsigned long long* chunk_sizes) {
    const std::size_t q{(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32};
    if (q >= num_groups * AC_GROUP_CHUNKS) {
        return;
    }
    const GroupExtent e{group_extent(q / AC_GROUP_CHUNKS, bw, bh, xg)};
    const std::size_t i{(q % AC_GROUP_CHUNKS) * 32 + threadIdx.x % 32};
    unsigned long long size{0};
    if (i < e.gbw * e.gbh * 3) {
        const std::size_t bidx{i / 3};
        size = item_sizes[((e.by0 + bidx / e.gbw) * bw + e.bx0 + bidx % e.gbw) * 3 + i % 3];
    }
    for (unsigned int d{16}; d > 0; d >>= 1) {
        size += __shfl_xor_sync(~0u, size, d);
    }
    if (threadIdx.x % 32 == 0) {
        chunk_sizes[q] = size;
    }
}

// Per group (launched with AC_GROUP_CHUNKS threads): the exclusive scan of the
// chunk sizes gives each chunk's token and raw-bit offsets within the group.
__global__ void ac_group_scan_kernel(const unsigned long long* chunk_sizes,
                                     unsigned long long* chunk_base,
                                     unsigned long long* group_sizes) {
    __shared__ unsigned long long sh[AC_GROUP_CHUNKS];
    const std::size_t chunk{static_cast<std::size_t>(blockIdx.x) * AC_GROUP_CHUNKS + threadIdx.x};
    unsigned long long total{0};
    chunk_base[chunk] = block_scan_exclusive(chunk_sizes[chunk], sh, total);
    if (threadIdx.x == 0) {
        group_sizes[blockIdx.x] = total;
    }
}

// One block scans the group sizes tile by tile, each group's raw bits rounded
// up to whole 64-bit words, into the token layout (see ac_token_layout).
__global__ void ac_group_layout_kernel(const unsigned long long* group_sizes,
                                       std::size_t num_groups, std::uint32_t* layout) {
    __shared__ unsigned long long sh[AC_LAYOUT_THREADS];
    std::uint32_t* const token_offset{layout};
    std::uint32_t* const raw_offset{layout + num_groups + 1};
    std::uint32_t* const raw_bits{layout + 2 * (num_groups + 1)};
    unsigned long long base{0};
    for (std::size_t g0{0}; g0 < num_groups; g0 += blockDim.x) {
        const std::size_t g{g0 + threadIdx.x};
        const unsigned long long size{g < num_groups ? group_sizes[g] : 0};
        const auto group_raw_bits{static_cast<std::uint32_t>(size >> 32)};
        unsigned long long total{0};
        const unsigned long long offset{
            base + block_scan_exclusive(
                       pack_sizes(static_cast<std::uint32_t>(size), (group_raw_bits + 63) / 64), sh,
                       total)};
        if (g < num_groups) {
            token_offset[g] = static_cast<std::uint32_t>(offset);
            raw_offset[g] = static_cast<std::uint32_t>(offset >> 32);
            raw_bits[g] = group_raw_bits;
        }
        base += total;
    }
    if (threadIdx.x == 0) {
        token_offset[num_groups] = static_cast<std::uint32_t>(base);
        raw_offset[num_groups] = static_cast<std::uint32_t>(base >> 32);
    }
}

// The pack kernel ORs raw bits into their words, so a batch's raw words start
// zeroed.
__global__ void zero_raw_words_kernel(unsigned long long* raw, const std::uint32_t* raw_offset,
                                      std::size_t group_begin, std::size_t group_end) {
    const std::size_t end{raw_offset[group_end]};
    const std::size_t stride{static_cast<std::size_t>(gridDim.x) * blockDim.x};
    for (std::size_t i{raw_offset[group_begin] + static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x};
         i < end; i += stride) {
        raw[i] = 0;
    }
}

__device__ void put_raw_bits(unsigned long long* raw, unsigned long long pos, std::uint32_t bits,
                             std::uint32_t n) {
    if (n == 0) {
        return;
    }
    const auto shift{static_cast<std::uint32_t>(pos & 63)};
    atomicOr(&raw[pos >> 6], static_cast<unsigned long long>(bits) << shift);
    if (shift + n > 64) {
        atomicOr(&raw[(pos >> 6) + 1], static_cast<unsigned long long>(bits) >> (64 - shift));
    }
}

__device__ std::uint16_t token_entry(const std::uint8_t* context_map, std::uint32_t context,
                                     std::uint32_t symbol) {
    return static_cast<std::uint16_t>(
        (static_cast<std::uint32_t>(__ldg(&context_map[context])) << bitstream::TOKEN_CLUSTER_SHIFT) |
        symbol);
}

// Writes the tokens of the chunk's items at their layout positions: each lane
// its item's non-zero-count token, then the warp each item's coefficient
// tokens, the active lanes in scan order.
__global__ void ac_pack_kernel(const std::int16_t* ac, const std::int8_t* acs,
                               const std::int32_t* nz_grid, const std::int32_t* nz_exact,
                               std::size_t bw, std::size_t bh, std::size_t nblocks, std::size_t xg,
                               std::size_t group_begin, std::size_t group_end,
                               const std::uint8_t* context_map,
                               const unsigned long long* item_sizes,
                               const unsigned long long* chunk_base, const std::uint32_t* layout,
                               std::size_t num_groups, std::uint16_t* entries,
                               unsigned long long* raw) {
    const std::size_t q{group_begin * AC_GROUP_CHUNKS +
                        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32};
    if (q >= group_end * AC_GROUP_CHUNKS) {
        return;
    }
    const std::size_t g{q / AC_GROUP_CHUNKS};
    const AcGroupCtx c{build_ac_ctx(g, ac, acs, nz_grid, nz_exact, bw, bh, nblocks, xg)};
    const unsigned int lane{threadIdx.x % 32};
    const unsigned int lanes_below{(1u << lane) - 1u};
    const std::size_t i{(q % AC_GROUP_CHUNKS) * 32 + lane};
    const AcLaneItem it{ac_lane_item(c, i)};
    const unsigned long long size{
        it.side != ACS_COVERED ? item_sizes[(it.by * bw + it.bx) * 3 + i % 3] : 0};
    const auto tokens{static_cast<std::uint32_t>(size)};
    const auto raw_bits{static_cast<std::uint32_t>(size >> 32)};
    const unsigned long long base{chunk_base[q]};
    const std::uint32_t token_pos{layout[g] + static_cast<std::uint32_t>(base) +
                                  warp_inclusive_sum(tokens, lane) - tokens};
    const unsigned long long raw_pos{static_cast<unsigned long long>(layout[num_groups + 1 + g]) *
                                         64 +
                                     (base >> 32) + warp_inclusive_sum(raw_bits, lane) - raw_bits};
    if (it.side != ACS_COVERED) {
        entries[token_pos] = token_entry(context_map, it.context, it.symbol);
        put_raw_bits(raw, raw_pos, it.bits, it.nbits);
    }
    unsigned int pending{__ballot_sync(~0u, it.side != ACS_COVERED && it.nzeros != 0)};
    while (pending != 0) {
        const int src{__ffs(pending) - 1};
        pending &= pending - 1;
        std::uint32_t next_token{__shfl_sync(~0u, token_pos + 1, src)};
        unsigned long long next_raw{__shfl_sync(~0u, raw_pos + it.nbits, src)};
        warp_item_coeff_tokens(
            c, it, src, lane,
            [&](bool active, std::uint32_t symbol, std::uint32_t nbits, std::uint32_t bits,
                std::uint32_t context) {
                const unsigned int mask{__ballot_sync(~0u, active)};
                if (active) {
                    entries[next_token + static_cast<std::uint32_t>(__popc(mask & lanes_below))] =
                        token_entry(context_map, context, symbol);
                }
                next_token += static_cast<std::uint32_t>(__popc(mask));
                if (__any_sync(~0u, active && nbits != 0)) {
                    const std::uint32_t n{active ? nbits : 0u};
                    const std::uint32_t end{warp_inclusive_sum(n, lane)};
                    put_raw_bits(raw, next_raw + end - n, bits, n);
                    next_raw += __shfl_sync(~0u, end, 31);
                }
            });
    }
}

// --- AC context clustering (see ac_clustering.h) -------------------------------

static_assert(AC_MAX_CLUSTERS == 32, "the assignment kernels take one cluster per lane");

struct ClusterScratch {
    std::uint64_t* masks;
    std::int64_t* distances;
    std::uint32_t* totals;
    std::uint32_t* cluster_hist;
    std::uint32_t* seeds;
    std::uint32_t* num_seeds;
    std::uint32_t* num_active;
    std::uint16_t* active;
    std::uint8_t* assignment;
};

constexpr std::size_t NUM_CONTEXTS{AC_NUM_CONTEXTS};

// Layout sized by ac_cluster_scratch_bytes; the 64-bit arrays come first so every
// array is naturally aligned.
ClusterScratch carve_cluster_scratch(std::uint8_t* scratch) {
    ClusterScratch s{};
    s.masks = reinterpret_cast<std::uint64_t*>(scratch);
    s.distances = reinterpret_cast<std::int64_t*>(s.masks + NUM_CONTEXTS);
    s.totals = reinterpret_cast<std::uint32_t*>(s.distances + NUM_CONTEXTS);
    s.cluster_hist = s.totals + NUM_CONTEXTS;
    s.seeds = s.cluster_hist + AC_MAX_CLUSTERS * AC_CONTEXT_ALPHABET;
    s.num_seeds = s.seeds + AC_MAX_CLUSTERS;
    s.num_active = s.num_seeds + 1;
    s.active = reinterpret_cast<std::uint16_t*>(s.num_active + 1);
    s.assignment = reinterpret_cast<std::uint8_t*>(s.active + NUM_CONTEXTS);
    return s;
}

__device__ AcHistogramRef context_ref(const std::uint32_t* context_histograms,
                                      const ClusterScratch& s, std::uint32_t context) {
    return AcHistogramRef{context_histograms + context * AC_CONTEXT_ALPHABET, s.totals[context],
                          s.masks[context]};
}

// The (value, index) pair of the lowest value, the lowest index among equal
// values, over the warp.
__device__ void warp_argmin(std::int64_t& value, std::uint32_t& index) {
    for (unsigned int d{16}; d > 0; d >>= 1) {
        const std::int64_t other_value{__shfl_xor_sync(~0u, value, d)};
        const std::uint32_t other_index{__shfl_xor_sync(~0u, index, d)};
        if (other_value < value || (other_value == value && other_index < index)) {
            value = other_value;
            index = other_index;
        }
    }
}

// The (value, index) pair of the highest value, the lowest index among equal
// values, over the thread block, returned to every thread. `sh_value` and
// `sh_index` hold 32 entries.
__device__ void block_argmax(std::int64_t& value, std::uint32_t& index, std::int64_t* sh_value,
                             std::uint32_t* sh_index) {
    const unsigned int lane{threadIdx.x % 32};
    const unsigned int warp{threadIdx.x / 32};
    const auto reduce = [&] {
        for (unsigned int d{16}; d > 0; d >>= 1) {
            const std::int64_t other_value{__shfl_xor_sync(~0u, value, d)};
            const std::uint32_t other_index{__shfl_xor_sync(~0u, index, d)};
            if (other_value > value || (other_value == value && other_index < index)) {
                value = other_value;
                index = other_index;
            }
        }
    };
    reduce();
    if (lane == 0) {
        sh_value[warp] = value;
        sh_index[warp] = index;
    }
    __syncthreads();
    if (warp == 0) {
        const bool has_warp{lane < blockDim.x / 32};
        value = has_warp ? sh_value[lane] : INT64_MIN;
        index = has_warp ? sh_index[lane] : UINT32_MAX;
        reduce();
        if (lane == 0) {
            sh_value[0] = value;
            sh_index[0] = index;
        }
    }
    __syncthreads();
    value = sh_value[0];
    index = sh_index[0];
    __syncthreads();
}

// Warp per context: its token total and the mask of its non-zero bins.
__global__ void context_stats_kernel(const std::uint32_t* context_histograms, ClusterScratch s) {
    static_assert(AC_CONTEXT_ALPHABET == 64);
    const std::size_t context{(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
                              32};
    if (context >= NUM_CONTEXTS) {
        return;
    }
    const unsigned int lane{threadIdx.x % 32};
    const std::uint32_t* bins{context_histograms + context * AC_CONTEXT_ALPHABET};
    const std::uint32_t lo{bins[lane]};
    const std::uint32_t hi{bins[lane + 32]};
    const std::uint32_t total{warp_sum(lo + hi)};
    const std::uint64_t mask{static_cast<std::uint64_t>(__ballot_sync(~0u, lo != 0)) |
                             (static_cast<std::uint64_t>(__ballot_sync(~0u, hi != 0)) << 32)};
    if (lane == 0) {
        s.totals[context] = total;
        s.masks[context] = mask;
    }
}

constexpr unsigned int CLUSTER_SEED_THREADS = 1024;

// One thread block: lists the contexts with tokens in context order, then picks
// the seeds as libjxl's FastClusterHistograms does. The first seed is the
// largest context; every round updates each context's merge cost to its nearest
// seed and takes the farthest context as the next seed, until AC_MAX_CLUSTERS
// seeds or the farthest is closer than AC_MIN_SEED_DISTANCE. A context at cost
// 0 (a seed, or one merging for free) is not considered again.
__global__ void cluster_seed_kernel(const std::uint32_t* context_histograms, ClusterScratch s) {
    __shared__ std::int32_t table[AC_LOG2_TABLE_SIZE + 1];
    __shared__ std::uint32_t seed_bins[AC_CONTEXT_ALPHABET];
    __shared__ unsigned long long scan[CLUSTER_SEED_THREADS];
    __shared__ std::int64_t sh_value[32];
    __shared__ std::uint32_t sh_index[32];
    for (unsigned int i{threadIdx.x}; i <= AC_LOG2_TABLE_SIZE; i += blockDim.x) {
        table[i] = AC_LOG2_TABLE[i];
    }

    std::uint32_t num_active{0};
    for (std::size_t base{0}; base < NUM_CONTEXTS; base += blockDim.x) {
        const std::size_t context{base + threadIdx.x};
        const bool flag{context < NUM_CONTEXTS && s.totals[context] != 0};
        unsigned long long count{0};
        const unsigned long long slot{block_scan_exclusive(flag ? 1 : 0, scan, count)};
        if (flag) {
            s.active[num_active + slot] = static_cast<std::uint16_t>(context);
        }
        num_active += static_cast<std::uint32_t>(count);
    }
    __syncthreads();
    std::uint32_t num_seeds{0};
    if (num_active != 0) {
        std::int64_t largest_total{-1};
        std::uint32_t largest{UINT32_MAX};
        for (std::uint32_t a{threadIdx.x}; a < num_active; a += blockDim.x) {
            const std::int64_t total{s.totals[s.active[a]]};
            if (total > largest_total) {
                largest_total = total;
                largest = a;
            }
            s.distances[a] = INT64_MAX;
        }
        block_argmax(largest_total, largest, sh_value, sh_index);
        for (;;) {
            const std::uint32_t seed{s.active[largest]};
            if (threadIdx.x == 0) {
                s.seeds[num_seeds] = seed;
                s.distances[largest] = 0;
            }
            if (threadIdx.x < AC_CONTEXT_ALPHABET) {
                seed_bins[threadIdx.x] = context_histograms[seed * AC_CONTEXT_ALPHABET + threadIdx.x];
            }
            ++num_seeds;
            if (num_seeds == AC_MAX_CLUSTERS) {
                break;
            }
            __syncthreads();
            const AcHistogramRef seed_ref{seed_bins, s.totals[seed], s.masks[seed]};
            std::int64_t farthest{-1};
            std::uint32_t farthest_slot{UINT32_MAX};
            for (std::uint32_t a{threadIdx.x}; a < num_active; a += blockDim.x) {
                std::int64_t distance{s.distances[a]};
                if (distance == 0) {
                    continue;
                }
                const std::int64_t cost{
                    ac_merge_cost(context_ref(context_histograms, s, s.active[a]), seed_ref, table)};
                distance = min(distance, max(cost, std::int64_t{0}));
                s.distances[a] = distance;
                if (distance > farthest) {
                    farthest = distance;
                    farthest_slot = a;
                }
            }
            block_argmax(farthest, farthest_slot, sh_value, sh_index);
            if (farthest < AC_MIN_SEED_DISTANCE) {
                break;
            }
            largest = farthest_slot;
        }
    }
    if (threadIdx.x == 0) {
        *s.num_seeds = num_seeds;
        *s.num_active = num_active;
    }
}

constexpr unsigned int CLUSTER_THREADS = 256;
constexpr unsigned int CLUSTER_CTAS = 64;

// The context-to-seed assignment and the Lloyd assignments: a warp per context
// with tokens, lane j weighing cluster j. The thread blocks stride over the
// contexts, so that each stages the clusters in shared memory once.
template <typename Cost>
__device__ void assign_contexts(const ClusterScratch& s, Cost cost) {
    const unsigned int lane{threadIdx.x % 32};
    const std::uint32_t num_active{*s.num_active};
    const std::uint32_t warps{gridDim.x * blockDim.x / 32};
    for (std::uint32_t a{(blockIdx.x * blockDim.x + threadIdx.x) / 32}; a < num_active; a += warps) {
        const std::uint32_t context{s.active[a]};
        std::int64_t value{cost(context, lane)};
        std::uint32_t index{lane};
        warp_argmin(value, index);
        if (lane == 0) {
            s.assignment[context] = static_cast<std::uint8_t>(index);
        }
    }
}

__global__ void cluster_assign_seeds_kernel(const std::uint32_t* context_histograms,
                                            ClusterScratch s) {
    __shared__ std::uint32_t bins[AC_MAX_CLUSTERS * AC_CONTEXT_ALPHABET];
    __shared__ AcHistogramRef seeds[AC_MAX_CLUSTERS];
    __shared__ std::uint32_t seed_contexts[AC_MAX_CLUSTERS];
    const std::uint32_t num_seeds{*s.num_seeds};
    for (std::uint32_t i{threadIdx.x}; i < num_seeds * AC_CONTEXT_ALPHABET; i += blockDim.x) {
        bins[i] = context_histograms[s.seeds[i / AC_CONTEXT_ALPHABET] * AC_CONTEXT_ALPHABET +
                                     i % AC_CONTEXT_ALPHABET];
    }
    for (std::uint32_t j{threadIdx.x}; j < num_seeds; j += blockDim.x) {
        const std::uint32_t seed{s.seeds[j]};
        seed_contexts[j] = seed;
        seeds[j] = AcHistogramRef{bins + j * AC_CONTEXT_ALPHABET, s.totals[seed], s.masks[seed]};
    }
    __syncthreads();
    assign_contexts(s, [&](std::uint32_t context, unsigned int j) -> std::int64_t {
        if (j >= num_seeds) {
            return INT64_MAX;
        }
        if (seed_contexts[j] == context) {
            return INT64_MIN;
        }
        return ac_merge_cost(context_ref(context_histograms, s, context), seeds[j], AC_LOG2_TABLE);
    });
}

// Warp per context with tokens: adds its histogram to its cluster's.
__global__ void cluster_histograms_kernel(const std::uint32_t* context_histograms,
                                          ClusterScratch s) {
    const unsigned int lane{threadIdx.x % 32};
    const std::uint32_t num_active{*s.num_active};
    const std::uint32_t warps{gridDim.x * blockDim.x / 32};
    for (std::uint32_t a{(blockIdx.x * blockDim.x + threadIdx.x) / 32}; a < num_active; a += warps) {
        const std::uint32_t context{s.active[a]};
        std::uint32_t* const cluster{s.cluster_hist + s.assignment[context] * AC_CONTEXT_ALPHABET};
        for (std::uint32_t sym{lane}; sym < AC_CONTEXT_ALPHABET; sym += 32) {
            const std::uint32_t count{context_histograms[context * AC_CONTEXT_ALPHABET + sym]};
            if (count != 0) {
                atomicAdd(&cluster[sym], count);
            }
        }
    }
}

// Lloyd assignment: every context moves to the cluster that codes it in the
// fewest bits, among those holding all of its symbols.
__global__ void cluster_assign_kernel(const std::uint32_t* context_histograms, ClusterScratch s) {
    __shared__ std::uint32_t bins[AC_MAX_CLUSTERS * AC_CONTEXT_ALPHABET];
    __shared__ AcHistogramRef clusters[AC_MAX_CLUSTERS];
    for (std::uint32_t i{threadIdx.x}; i < AC_MAX_CLUSTERS * AC_CONTEXT_ALPHABET; i += blockDim.x) {
        bins[i] = s.cluster_hist[i];
    }
    __syncthreads();
    const unsigned int lane{threadIdx.x % 32};
    for (std::uint32_t j{threadIdx.x / 32}; j < AC_MAX_CLUSTERS; j += blockDim.x / 32) {
        const std::uint32_t lo{bins[j * AC_CONTEXT_ALPHABET + lane]};
        const std::uint32_t hi{bins[j * AC_CONTEXT_ALPHABET + lane + 32]};
        const std::uint32_t total{warp_sum(lo + hi)};
        const std::uint64_t mask{static_cast<std::uint64_t>(__ballot_sync(~0u, lo != 0)) |
                                 (static_cast<std::uint64_t>(__ballot_sync(~0u, hi != 0)) << 32)};
        if (lane == 0) {
            clusters[j] = AcHistogramRef{bins + j * AC_CONTEXT_ALPHABET, total, mask};
        }
    }
    __syncthreads();
    assign_contexts(s, [&](std::uint32_t context, unsigned int j) -> std::int64_t {
        const AcHistogramRef a{context_ref(context_histograms, s, context)};
        if (clusters[j].total == 0 || (a.mask & ~clusters[j].mask) != 0) {
            return INT64_MAX;
        }
        return ac_coding_cost(a, clusters[j], AC_LOG2_TABLE);
    });
}

// Inclusive maximum scan of `v` (>= -1) over the thread block, with the block's
// maximum in `total`; `sh` holds blockDim.x entries.
__device__ int block_scan_max(int v, int* sh, int& total) {
    const unsigned int tid{threadIdx.x};
    sh[tid] = v;
    __syncthreads();
    for (unsigned int offset{1}; offset < blockDim.x; offset <<= 1) {
        const int other{tid >= offset ? sh[tid - offset] : -1};
        __syncthreads();
        sh[tid] = max(sh[tid], other);
        __syncthreads();
    }
    total = sh[blockDim.x - 1];
    const int inclusive{sh[tid]};
    __syncthreads();
    return inclusive;
}

constexpr unsigned int CLUSTER_FINISH_THREADS = 1024;

// One thread block: numbers the non-empty clusters by their first context and
// writes the context map and the renumbered histograms.
__global__ void cluster_finish_kernel(ClusterScratch s, AcClustering* out) {
    __shared__ std::uint32_t first[AC_MAX_CLUSTERS];
    __shared__ std::uint32_t rank[AC_MAX_CLUSTERS];
    __shared__ std::uint32_t inverse[AC_MAX_CLUSTERS];
    __shared__ int scan[CLUSTER_FINISH_THREADS];
    if (threadIdx.x < AC_MAX_CLUSTERS) {
        first[threadIdx.x] = UINT32_MAX;
    }
    __syncthreads();
    const std::uint32_t num_active{*s.num_active};
    for (std::uint32_t a{threadIdx.x}; a < num_active; a += blockDim.x) {
        const std::uint32_t context{s.active[a]};
        atomicMin(&first[s.assignment[context]], context);
    }
    __syncthreads();
    std::uint32_t num_clusters{0};
    for (std::uint32_t j{0}; j < AC_MAX_CLUSTERS; ++j) {
        num_clusters += first[j] != UINT32_MAX ? 1 : 0;
    }
    if (threadIdx.x < AC_MAX_CLUSTERS && first[threadIdx.x] != UINT32_MAX) {
        std::uint32_t r{0};
        for (std::uint32_t j{0}; j < AC_MAX_CLUSTERS; ++j) {
            r += first[j] < first[threadIdx.x] ? 1 : 0;
        }
        rank[threadIdx.x] = r;
        inverse[r] = threadIdx.x;
    }
    __syncthreads();

    // The last context with tokens up to each context.
    int last_active{-1};
    for (std::size_t base{0}; base < NUM_CONTEXTS; base += blockDim.x) {
        const std::size_t context{base + threadIdx.x};
        const bool has_tokens{context < NUM_CONTEXTS && s.totals[context] != 0};
        int tile_last{-1};
        const int last{max(last_active, block_scan_max(has_tokens ? static_cast<int>(context) : -1,
                                                       scan, tile_last))};
        if (context < NUM_CONTEXTS) {
            out->context_map[context] = static_cast<std::uint8_t>(
                last < 0 ? 0 : rank[s.assignment[static_cast<std::size_t>(last)]]);
        }
        last_active = max(last_active, tile_last);
    }
    for (std::uint32_t i{threadIdx.x}; i < AC_MAX_CLUSTERS * AC_CONTEXT_ALPHABET; i += blockDim.x) {
        const std::size_t r{i / AC_CONTEXT_ALPHABET};
        out->histograms[i] =
            r < num_clusters
                ? s.cluster_hist[inverse[r] * AC_CONTEXT_ALPHABET + i % AC_CONTEXT_ALPHABET]
                : 0;
    }
    if (threadIdx.x == 0) {
        out->num_clusters = max(num_clusters, 1u);
    }
}

struct DcEncodeScratch {
    unsigned long long* chunk_bits{nullptr};
    unsigned long long* chunk_base{nullptr};
    std::uint32_t* run_offsets{nullptr};
};

// Layout sized by dc_encode_scratch_bytes; the 64-bit arrays come first so every
// array is naturally aligned.
DcEncodeScratch carve_dc_encode_scratch(std::uint8_t* scratch, std::size_t num_groups) {
    const std::size_t chunks{num_groups * DC_CHUNKS};
    auto* const chunk_bits{reinterpret_cast<unsigned long long*>(scratch)};
    return DcEncodeScratch{
        .chunk_bits = chunk_bits,
        .chunk_base = chunk_bits + chunks,
        .run_offsets = reinterpret_cast<std::uint32_t*>(chunk_bits + 2 * chunks)};
}

}  // namespace

std::size_t ac_num_groups(std::size_t width, std::size_t height) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xg{(bw + AC_GROUP_BLOCKS - 1) / AC_GROUP_BLOCKS};
    const std::size_t yg{(bh + AC_GROUP_BLOCKS - 1) / AC_GROUP_BLOCKS};
    return xg * yg;
}

bool ac_build_histogram(const std::int16_t* ac, const std::int8_t* acs, const std::int32_t* nz_grid,
                        const std::int32_t* nz_exact, std::size_t width, std::size_t height,
                        std::uint32_t* context_histograms, std::uint64_t* item_sizes,
                        cudaStream_t stream) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t nblocks{bw * bh};
    if (cudaMemsetAsync(context_histograms, 0,
                        AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET * sizeof(std::uint32_t),
                        stream) != cudaSuccess) {
        return false;
    }
    const std::size_t total{3 * bw * bh};
    const unsigned int blocks{
        static_cast<unsigned int>((total + HIST_ITEMS_PER_CTA - 1) / HIST_ITEMS_PER_CTA)};
    histogram_kernel<<<blocks, HIST_THREADS, 0, stream>>>(
        ac, acs, nz_grid, nz_exact, bw, bh, nblocks, context_histograms,
        reinterpret_cast<unsigned long long*>(item_sizes));
    return cudaGetLastError() == cudaSuccess;
}

std::size_t ac_token_chunks(std::size_t width, std::size_t height) {
    return ac_num_groups(width, height) * AC_GROUP_CHUNKS;
}

std::size_t ac_token_layout_scratch_bytes(std::size_t width, std::size_t height) {
    return (ac_token_chunks(width, height) + ac_num_groups(width, height)) *
           sizeof(unsigned long long);
}

std::size_t ac_max_tokens(std::size_t width, std::size_t height) {
    // Per channel, a first-block of side N codes its count token and at most its
    // N * N - (N / 8)^2 coefficients: at most 64 tokens per covered 8x8 block.
    return 3 * (width / 8) * (height / 8) * COEFFS_PER_BLOCK;
}

std::size_t ac_max_raw_words(std::size_t width, std::size_t height) {
    return ac_max_tokens(width, height) * AC_MAX_TOKEN_RAW_BITS / 64 + ac_num_groups(width, height);
}

bool ac_token_layout(const std::uint64_t* item_sizes, std::size_t width, std::size_t height,
                     std::uint8_t* scratch, std::uint64_t* chunk_base, std::uint32_t* layout,
                     cudaStream_t stream) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xg{(bw + AC_GROUP_BLOCKS - 1) / AC_GROUP_BLOCKS};
    const std::size_t num_groups{ac_num_groups(width, height)};
    const std::size_t num_chunks{num_groups * AC_GROUP_CHUNKS};
    auto* const chunk_sizes{reinterpret_cast<unsigned long long*>(scratch)};
    unsigned long long* const group_sizes{chunk_sizes + num_chunks};
    const std::size_t chunks_per_cta{AC_CHUNK_THREADS / 32};
    ac_chunk_sizes_kernel<<<static_cast<unsigned int>((num_chunks + chunks_per_cta - 1) /
                                                      chunks_per_cta),
                            AC_CHUNK_THREADS, 0, stream>>>(
        reinterpret_cast<const unsigned long long*>(item_sizes), bw, bh, xg, num_groups,
        chunk_sizes);
    ac_group_scan_kernel<<<static_cast<unsigned int>(num_groups), AC_GROUP_CHUNKS, 0, stream>>>(
        chunk_sizes, reinterpret_cast<unsigned long long*>(chunk_base), group_sizes);
    ac_group_layout_kernel<<<1, AC_LAYOUT_THREADS, 0, stream>>>(group_sizes, num_groups, layout);
    return cudaGetLastError() == cudaSuccess;
}

bool ac_pack_tokens(const std::int16_t* ac, const std::int8_t* acs, const std::int32_t* nz_grid,
                    const std::int32_t* nz_exact, std::size_t width, std::size_t height,
                    const std::uint8_t* context_map, const std::uint64_t* item_sizes,
                    const std::uint64_t* chunk_base, const std::uint32_t* layout,
                    std::size_t group_begin, std::size_t group_end, std::uint16_t* entries,
                    std::uint64_t* raw, cudaStream_t stream) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xg{(bw + AC_GROUP_BLOCKS - 1) / AC_GROUP_BLOCKS};
    const std::size_t num_groups{ac_num_groups(width, height)};
    auto* const raw_words{reinterpret_cast<unsigned long long*>(raw)};
    zero_raw_words_kernel<<<ZERO_FILL_BLOCKS, ZERO_FILL_THREADS, 0, stream>>>(
        raw_words, layout + num_groups + 1, group_begin, group_end);
    const std::size_t num_chunks{(group_end - group_begin) * AC_GROUP_CHUNKS};
    const std::size_t chunks_per_cta{AC_CHUNK_THREADS / 32};
    ac_pack_kernel<<<static_cast<unsigned int>((num_chunks + chunks_per_cta - 1) / chunks_per_cta),
                     AC_CHUNK_THREADS, 0, stream>>>(
        ac, acs, nz_grid, nz_exact, bw, bh, bw * bh, xg, group_begin, group_end, context_map,
        reinterpret_cast<const unsigned long long*>(item_sizes),
        reinterpret_cast<const unsigned long long*>(chunk_base), layout, num_groups, entries,
        raw_words);
    return cudaGetLastError() == cudaSuccess;
}

std::size_t ac_cluster_scratch_bytes() {
    return NUM_CONTEXTS * (sizeof(std::uint64_t) + sizeof(std::int64_t) + sizeof(std::uint32_t) +
                           sizeof(std::uint16_t) + sizeof(std::uint8_t)) +
           (AC_MAX_CLUSTERS * AC_CONTEXT_ALPHABET + AC_MAX_CLUSTERS + 2) * sizeof(std::uint32_t);
}

bool ac_cluster_contexts(const std::uint32_t* context_histograms, std::uint8_t* scratch,
                         AcClustering* clustering, cudaStream_t stream) {
    ensure_constants();
    const ClusterScratch s{carve_cluster_scratch(scratch)};
    context_stats_kernel<<<static_cast<unsigned int>(
                               (NUM_CONTEXTS * 32 + CLUSTER_THREADS - 1) / CLUSTER_THREADS),
                           CLUSTER_THREADS, 0, stream>>>(context_histograms, s);
    cluster_seed_kernel<<<1, CLUSTER_SEED_THREADS, 0, stream>>>(context_histograms, s);
    cluster_assign_seeds_kernel<<<CLUSTER_CTAS, CLUSTER_THREADS, 0, stream>>>(context_histograms,
                                                                                s);
    for (int it{0}; it <= AC_CLUSTER_LLOYD_ITERATIONS; ++it) {
        if (cudaMemsetAsync(s.cluster_hist, 0,
                            AC_MAX_CLUSTERS * AC_CONTEXT_ALPHABET * sizeof(std::uint32_t),
                            stream) != cudaSuccess) {
            return false;
        }
        cluster_histograms_kernel<<<CLUSTER_CTAS, CLUSTER_THREADS, 0, stream>>>(context_histograms,
                                                                                  s);
        if (it < AC_CLUSTER_LLOYD_ITERATIONS) {
            cluster_assign_kernel<<<CLUSTER_CTAS, CLUSTER_THREADS, 0, stream>>>(context_histograms,
                                                                                  s);
        }
    }
    cluster_finish_kernel<<<1, CLUSTER_FINISH_THREADS, 0, stream>>>(s, clustering);
    return cudaGetLastError() == cudaSuccess;
}

std::size_t dc_num_groups(std::size_t width, std::size_t height) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t ydg{(bh + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    return xdg * ydg;
}

std::size_t dc_encode_scratch_bytes(std::size_t width, std::size_t height) {
    const std::size_t chunks{dc_num_groups(width, height) * DC_CHUNKS};
    return 2 * chunks * sizeof(unsigned long long) +
           chunks * DC_EMIT_THREADS * sizeof(std::uint32_t);
}

std::size_t dc_first_blocks_scratch_bytes(std::size_t width, std::size_t height) {
    return dc_num_groups(width, height) * DC_GROUP_BLOCKS * sizeof(std::uint32_t);
}

bool dc_first_blocks(const std::int8_t* acs, std::size_t width, std::size_t height,
                     std::uint8_t* scratch, std::size_t* positions, std::size_t* counts,
                     cudaStream_t stream) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    auto* const row_counts{reinterpret_cast<std::uint32_t*>(scratch)};
    const dim3 grid{static_cast<unsigned int>(bh), static_cast<unsigned int>(xdg)};
    first_block_row_count_kernel<<<grid, FIRST_BLOCK_THREADS, 0, stream>>>(acs, bw, xdg,
                                                                           row_counts);
    first_block_scatter_kernel<<<grid, FIRST_BLOCK_THREADS, 0, stream>>>(
        acs, bw, bh, xdg, row_counts, positions, counts);
    return cudaGetLastError() == cudaSuccess;
}

bool dc_build_histograms(const std::int32_t* dc, std::size_t width, std::size_t height,
                         std::uint32_t* histograms, cudaStream_t stream) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t nblocks{bw * bh};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t num_groups{dc_num_groups(width, height)};
    if (cudaMemsetAsync(histograms, 0, num_groups * AC_HISTOGRAM_SIZE * sizeof(std::uint32_t),
                        stream) != cudaSuccess) {
        return false;
    }
    dc_histogram_kernel<<<group_hist_grid(DC_HIST_ROWS, bh, xdg), GROUP_HIST_THREADS, 0, stream>>>(
        dc, bw, bh, nblocks, xdg, histograms);
    return cudaGetLastError() == cudaSuccess;
}

bool acmeta_build_histograms(const std::int32_t* quant_field, const std::int8_t* acs,
                             const std::int8_t* ytox_map, const std::int8_t* ytob_map,
                             std::size_t width, std::size_t height, std::uint32_t* histograms,
                             cudaStream_t stream) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t cmw{(bw + 7) / 8};
    const std::size_t num_groups{dc_num_groups(width, height)};
    if (cudaMemsetAsync(histograms, 0, num_groups * AC_HISTOGRAM_SIZE * sizeof(std::uint32_t),
                        stream) != cudaSuccess) {
        return false;
    }
    acmeta_histogram_kernel<<<group_hist_grid(ACMETA_HIST_ROWS, bh, xdg), GROUP_HIST_THREADS, 0,
                              stream>>>(quant_field, acs, ytox_map, ytob_map, bw, bh, cmw, xdg,
                                        histograms);
    return cudaGetLastError() == cudaSuccess;
}

bool dc_encode_groups(const std::int32_t* dc, std::size_t width, std::size_t height,
                      const std::int32_t* quant_field, const std::int8_t* acs,
                      const std::int8_t* ytox_map, const std::int8_t* ytob_map,
                      const std::size_t* first_block_pos, const std::size_t* first_block_counts,
                      const std::uint8_t* dc_depth, const std::uint16_t* dc_bits,
                      const std::uint8_t* acmeta_depth, const std::uint16_t* acmeta_bits,
                      const std::uint8_t* blob_pre, const std::uint32_t* blob_pre_off,
                      const std::uint32_t* blob_pre_bits, const std::uint8_t* blob_mid,
                      const std::uint32_t* blob_mid_off, const std::uint32_t* blob_mid_bits,
                      std::uint8_t* scratch, std::uint8_t* out, std::size_t out_capacity,
                      std::uint32_t* group_sizes, std::uint32_t* group_offsets,
                      cudaStream_t stream) {
    ensure_constants();
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t nblocks{bw * bh};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t cmw{(bw + 7) / 8};
    const std::size_t num_groups{dc_num_groups(width, height)};

    const DcEncodeScratch s{carve_dc_encode_scratch(scratch, num_groups)};

    const dim3 chunk_grid{static_cast<unsigned int>(num_groups), DC_CHUNKS};
    dc_chunk_size_kernel<<<chunk_grid, DC_EMIT_THREADS, 0, stream>>>(
        dc, bw, bh, nblocks, xdg, num_groups, quant_field, acs, ytox_map, ytob_map, cmw,
        first_block_pos, first_block_counts, dc_depth, dc_bits, acmeta_depth, acmeta_bits, blob_pre,
        blob_pre_off, blob_pre_bits, blob_mid, blob_mid_off, blob_mid_bits, s.run_offsets,
        s.chunk_bits);
    dc_chunk_scan_kernel<<<1, DC_CHUNKS, 0, stream>>>(num_groups, s.chunk_bits, s.chunk_base,
                                                      group_sizes, group_offsets);
    zero_section_kernel<<<ZERO_FILL_BLOCKS, ZERO_FILL_THREADS, 0, stream>>>(
        reinterpret_cast<unsigned int*>(out), group_offsets, group_sizes, num_groups, out_capacity);
    dc_chunk_emit_kernel<<<chunk_grid, DC_EMIT_THREADS, 0, stream>>>(
        dc, bw, bh, nblocks, xdg, num_groups, quant_field, acs, ytox_map, ytob_map, cmw,
        first_block_pos, first_block_counts, dc_depth, dc_bits, acmeta_depth, acmeta_bits, blob_pre,
        blob_pre_off, blob_pre_bits, blob_mid, blob_mid_off, blob_mid_bits, s.run_offsets,
        s.chunk_base, out, out_capacity, group_sizes, group_offsets);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace cujpegxl
