// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "prefix_build.h"

#include <cassert>

#include <cuda_runtime.h>

#include "dc_predict.h"
#include "src/bitstream/bit_sink.h"
#include "src/bitstream/frame_assembly.h"
#include "src/bitstream/serialize_impl.h"
#include "vardct_layout.h"
#include "warp_scan.cuh"

namespace cujpegxl {
namespace {

using bitstream::HISTOGRAM_STRIDE;
using bitstream::HybridUintConfig;
namespace detail = bitstream::detail;

constexpr unsigned int FULL_MASK{~0u};

// Each prefix code (DC or AcMetadata group histogram) is built and described by
// one warp, as a single-warp block.
constexpr unsigned int CODE_THREADS{32};

// Code lengths of a length-limited (<= 15 bits) code are below this.
constexpr std::uint32_t DEPTH_VALUES{16};

constexpr std::size_t MAX_TREE_NODES{2 * detail::MAX_ALPHABET + 4};

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

__device__ unsigned int lanemask_lt(unsigned int lane) {
    return (1u << lane) - 1u;
}

__device__ std::uint32_t warp_max(std::uint32_t v) {
    for (unsigned int d{16}; d != 0; d >>= 1) {
        v = max(v, __shfl_xor_sync(FULL_MASK, v, d));
    }
    return v;
}

// LSB-first bit writer into a zeroed shared word buffer from an arbitrary bit
// position. Bits are composed with atomicOr, so writers over disjoint bit ranges
// may share boundary words and run concurrently. When all lanes of a warp run
// the same serial field writes, only the lane with `emit` set stores bits while
// every lane advances `pos`, which thus stays warp-uniform.
struct SharedBitWriter {
    unsigned int* words;
    std::uint32_t pos;
    bool emit;

    __device__ void write(std::size_t n_bits, std::uint64_t bits) {
        assert(n_bits <= 32);
        assert((bits >> n_bits) == 0);
        if (emit && n_bits != 0) {
            const std::uint64_t shifted{bits << (pos & 31u)};
            atomicOr(&words[pos >> 5], static_cast<unsigned int>(shifted));
            const auto high{static_cast<unsigned int>(shifted >> 32)};
            if (high != 0) {
                atomicOr(&words[(pos >> 5) + 1], high);
            }
        }
        pos += static_cast<std::uint32_t>(n_bits);
    }
};

struct TreeScratch {
    std::uint32_t leaf_count[detail::MAX_ALPHABET];
    std::uint16_t leaf_symbol[detail::MAX_ALPHABET];
    std::uint32_t node_count[MAX_TREE_NODES];
    std::uint16_t node_symbol[detail::MAX_ALPHABET];
    std::uint16_t parent[MAX_TREE_NODES];
};

// The code-length code of a complex code description.
struct CodeLengthCode {
    std::uint32_t hist[detail::CODE_LENGTH_CODES];
    std::uint8_t depth[detail::CODE_LENGTH_CODES];
    std::uint16_t bits[detail::CODE_LENGTH_CODES];
};

struct HistogramCode {
    std::uint32_t hist[HISTOGRAM_STRIDE];
    std::uint8_t depth[HISTOGRAM_STRIDE];
    std::uint16_t bits[HISTOGRAM_STRIDE];
};

struct CodeScratch {
    HistogramCode code;
    CodeLengthCode code_length;
    TreeScratch tree;
};

// Stages a HISTOGRAM_STRIDE histogram into `code` and zeroes its prefix code.
__device__ void load_histogram(const std::uint32_t* hist, HistogramCode& code, unsigned int lane) {
    for (std::size_t i{lane}; i < HISTOGRAM_STRIDE; i += CODE_THREADS) {
        code.hist[i] = hist[i];
        code.depth[i] = 0;
        code.bits[i] = 0;
    }
    __syncwarp();
}

__device__ void store_code(const HistogramCode& code, std::uint8_t* depth, std::uint16_t* bits,
                           unsigned int lane) {
    for (std::size_t i{lane}; i < HISTOGRAM_STRIDE; i += CODE_THREADS) {
        depth[i] = code.depth[i];
        bits[i] = code.bits[i];
    }
}

// Warp-uniform statistics of the used symbols of hist[0, length).
struct HistogramSummary {
    // Trailing-trimmed alphabet size, 1 if empty.
    std::uint32_t alphabet;
    std::uint32_t used;
    // The first min(used, 4) used symbols, the simple-code form's symbols.
    std::size_t first[4];
};

__device__ HistogramSummary summarize_histogram(const std::uint32_t* hist, std::uint32_t length,
                                                unsigned int lane) {
    HistogramSummary s{1, 0, {0, 0, 0, 0}};
    std::uint32_t found{0};
    for (std::uint32_t base{0}; base < length; base += 32) {
        const std::uint32_t i{base + lane};
        const unsigned int nonzero{__ballot_sync(FULL_MASK, i < length && hist[i] != 0)};
        if (nonzero != 0) {
            s.alphabet = base + 32 - __clz(nonzero);
        }
        for (unsigned int rest{nonzero}; rest != 0 && found < 4; rest &= rest - 1) {
            s.first[found++] = base + __ffs(rest) - 1;
        }
        s.used += __popc(nonzero);
    }
    return s;
}

// Returns, in lane v < DEPTH_VALUES, how many of values[0, length) equal v.
__device__ std::uint32_t count_depth_values(const std::uint8_t* values, std::uint32_t length,
                                            unsigned int lane) {
    std::uint32_t count{0};
    for (std::uint32_t base{0}; base < length; base += 32) {
        const std::uint32_t i{base + lane};
        const std::uint32_t v{i < length ? values[i] : DEPTH_VALUES};
        for (std::uint32_t d{0}; d < DEPTH_VALUES; ++d) {
            const unsigned int same{__ballot_sync(FULL_MASK, v == d)};
            if (lane == d) {
                count += __popc(same);
            }
        }
    }
    return count;
}

// Warp-cooperative detail::create_huffman_tree, producing the same depths: the
// stable sort is a rank sort (ties keep leaf order), one lane runs the two-queue
// merge recording parent links, and each leaf's depth is its parent-chain length.
// `depth` must be zeroed over [0, length).
__device__ void warp_create_huffman_tree(const std::uint32_t* data, std::uint32_t length,
                                         std::uint32_t tree_limit, std::uint8_t* depth,
                                         TreeScratch& t, unsigned int lane) {
    // Leaves in descending symbol order, as libjxl collects them.
    std::uint32_t n{0};
    for (std::uint32_t base{0}; base < length; base += 32) {
        const std::uint32_t r{base + lane};
        const std::uint32_t count{r < length ? data[length - 1 - r] : 0};
        const unsigned int nonzero{__ballot_sync(FULL_MASK, count != 0)};
        if (count != 0) {
            const std::uint32_t slot{n + __popc(nonzero & lanemask_lt(lane))};
            t.leaf_count[slot] = count;
            t.leaf_symbol[slot] = static_cast<std::uint16_t>(length - 1 - r);
        }
        n += __popc(nonzero);
    }
    __syncwarp();
    if (n == 1) {
        if (lane == 0) {
            depth[t.leaf_symbol[0]] = 1;
        }
        __syncwarp();
        return;
    }

    const std::uint32_t root{2 * n - 1};
    for (std::uint32_t count_limit{1};; count_limit *= 2) {
        for (std::uint32_t j{lane}; j < n; j += 32) {
            const std::uint32_t cj{max(t.leaf_count[j], count_limit - 1)};
            std::uint32_t rank{0};
            for (std::uint32_t k{0}; k < n; ++k) {
                const std::uint32_t ck{max(t.leaf_count[k], count_limit - 1)};
                rank += ck < cj || (ck == cj && k < j);
            }
            t.node_count[rank] = cj;
            t.node_symbol[rank] = t.leaf_symbol[j];
        }
        __syncwarp();

        if (lane == 0) {
            constexpr std::uint32_t SENTINEL{0xFFFFFFFFu};
            t.node_count[n] = SENTINEL;
            t.node_count[n + 1] = SENTINEL;
            std::uint32_t size{n + 2};
            std::uint32_t i{0};
            std::uint32_t j{n + 1};
            for (std::uint32_t k{n - 1}; k != 0; --k) {
                const std::uint32_t left{t.node_count[i] <= t.node_count[j] ? i++ : j++};
                const std::uint32_t right{t.node_count[i] <= t.node_count[j] ? i++ : j++};
                const std::uint32_t parent{size - 1};
                t.node_count[parent] = t.node_count[left] + t.node_count[right];
                t.parent[left] = static_cast<std::uint16_t>(parent);
                t.parent[right] = static_cast<std::uint16_t>(parent);
                t.node_count[size++] = SENTINEL;
            }
        }
        __syncwarp();

        std::uint32_t max_depth{0};
        for (std::uint32_t k{lane}; k < n; k += 32) {
            std::uint32_t d{0};
            for (std::uint32_t node{k}; node != root; node = t.parent[node]) {
                ++d;
            }
            depth[t.node_symbol[k]] = static_cast<std::uint8_t>(d);
            max_depth = max(max_depth, d);
        }
        max_depth = warp_max(max_depth);
        __syncwarp();
        if (max_depth <= tree_limit) {
            return;
        }
    }
}

// Warp-cooperative detail::convert_bit_depths_to_symbols: a symbol's canonical
// code is its depth's first code plus its rank among the symbols of that depth.
__device__ void warp_convert_bit_depths_to_symbols(const std::uint8_t* depth, std::uint32_t length,
                                                   std::uint16_t* bits, unsigned int lane) {
    std::uint32_t bl_count{count_depth_values(depth, length, lane)};
    if (lane == 0) {
        bl_count = 0;
    }
    std::uint32_t code{0};
    std::uint32_t next_code{0};
    for (std::uint32_t d{1}; d < DEPTH_VALUES; ++d) {
        code = (code + __shfl_sync(FULL_MASK, bl_count, d - 1)) << 1;
        if (lane == d) {
            next_code = code;
        }
    }
    for (std::uint32_t base{0}; base < length; base += 32) {
        const std::uint32_t i{base + lane};
        const std::uint32_t own{i < length ? depth[i] : 0u};
        for (std::uint32_t d{1}; d < DEPTH_VALUES; ++d) {
            const unsigned int same{__ballot_sync(FULL_MASK, own == d)};
            const std::uint32_t first{__shfl_sync(FULL_MASK, next_code, d)};
            if (own == d) {
                const auto symbol{
                    static_cast<std::uint16_t>(first + __popc(same & lanemask_lt(lane)))};
                bits[i] = static_cast<std::uint16_t>(__brev(symbol) >> (32 - d));
            }
            if (lane == d) {
                next_code += __popc(same);
            }
        }
    }
    __syncwarp();
}

// Warp-cooperative detail::store_huffman_tree. write_huffman_tree emits exactly
// one code-length token per symbol up to the last used one (the token being the
// depth itself, never a repeat code), so the tokens are written in parallel at
// warp-scanned bit offsets.
__device__ void warp_store_huffman_tree(const std::uint8_t* depth, std::uint32_t num,
                                        SharedBitWriter& w, CodeLengthCode& cl, TreeScratch& t,
                                        unsigned int lane) {
    std::uint32_t tokens{0};
    for (std::uint32_t base{0}; base < num; base += 32) {
        const std::uint32_t i{base + lane};
        const unsigned int nonzero{__ballot_sync(FULL_MASK, i < num && depth[i] != 0)};
        if (nonzero != 0) {
            tokens = base + 32 - __clz(nonzero);
        }
    }

    const std::uint32_t token_count{count_depth_values(depth, tokens, lane)};
    if (lane < detail::CODE_LENGTH_CODES) {
        cl.hist[lane] = lane < DEPTH_VALUES ? token_count : 0;
        cl.depth[lane] = 0;
        cl.bits[lane] = 0;
    }
    const unsigned int used_codes{
        __ballot_sync(FULL_MASK, lane < DEPTH_VALUES && token_count != 0)};
    const int num_codes{__popc(used_codes) > 1 ? 2 : 1};
    const int code{__ffs(used_codes) - 1};
    __syncwarp();

    warp_create_huffman_tree(cl.hist, detail::CODE_LENGTH_CODES, 5, cl.depth, t, lane);
    warp_convert_bit_depths_to_symbols(cl.depth, detail::CODE_LENGTH_CODES, cl.bits, lane);
    detail::store_huffman_tree_of_huffman_tree(num_codes, cl.depth, w);
    __syncwarp();
    if (num_codes == 1 && lane == 0) {
        cl.depth[code] = 0;
    }
    __syncwarp();

    for (std::uint32_t base{0}; base < tokens; base += 32) {
        const std::uint32_t i{base + lane};
        const bool active{i < tokens};
        const std::uint32_t token{active ? depth[i] : 0u};
        const std::uint32_t nbits{active ? cl.depth[token] : 0u};
        const std::uint32_t end{warp_inclusive_sum(nbits, lane)};
        SharedBitWriter{w.words, w.pos + end - nbits, true}.write(nbits,
                                                                  active ? cl.bits[token] : 0u);
        w.pos += __shfl_sync(FULL_MASK, end, 31);
    }
}

// Warp-cooperative detail::build_huffman_code + detail::store_huffman_code for
// code.hist[0, length), whose used symbols `s` describes.
__device__ void warp_build_and_store_huffman_code(HistogramCode& code, std::uint32_t length,
                                                  const HistogramSummary& s, SharedBitWriter& w,
                                                  CodeLengthCode& cl, TreeScratch& t,
                                                  unsigned int lane) {
    std::size_t max_bits{0};
    for (std::size_t v{length - 1}; v != 0; v >>= 1) {
        ++max_bits;
    }
    if (s.used <= 1) {
        w.write(4, 1);
        w.write(max_bits, s.first[0]);
        return;
    }
    warp_create_huffman_tree(code.hist, length, 15, code.depth, t, lane);
    warp_convert_bit_depths_to_symbols(code.depth, length, code.bits, lane);
    if (s.used <= 4) {
        std::size_t symbols[4]{s.first[0], s.first[1], s.first[2], s.first[3]};
        detail::store_simple_huffman_tree(code.depth, symbols, s.used, max_bits, w);
    } else {
        warp_store_huffman_tree(code.depth, length, w, cl, t, lane);
    }
}

// The code half of detail::write_prefix_histograms, warp-cooperative.
__device__ void warp_write_prefix_code(HistogramCode& code, std::uint32_t length,
                                       SharedBitWriter& w, CodeLengthCode& cl, TreeScratch& t,
                                       unsigned int lane) {
    if (length > 1) {
        const HistogramSummary s{summarize_histogram(code.hist, length, lane)};
        warp_build_and_store_huffman_code(code, length, s, w, cl, t, lane);
    }
}

// Warp-cooperative detail::write_prefix_histograms.
__device__ void warp_write_prefix_histograms(HistogramCode& code, std::uint32_t length,
                                             std::size_t num_contexts, SharedBitWriter& w,
                                             CodeLengthCode& cl, TreeScratch& t,
                                             unsigned int lane) {
    detail::write_prefix_histogram_head(w, length, num_contexts, HybridUintConfig{});
    warp_write_prefix_code(code, length, w, cl, t, lane);
}

struct DcBlobScratch {
    CodeScratch data;
    HistogramCode tree;
    unsigned int words[DC_BLOB_STRIDE / 4];
};

static_assert(detail::TREE_ALPHABET <= CODE_THREADS);

// Warp-cooperative detail::write_modular_tree.
__device__ void warp_write_modular_tree(SharedBitWriter& w, const detail::TreeTokens& tree,
                                        DcBlobScratch& s, unsigned int lane) {
    detail::write_group_header(w);
    std::uint32_t tree_count{0};
    std::uint32_t tree_len{1};
    for (std::size_t i{0}; i < tree.count; ++i) {
        tree_count += tree.values[i] == lane;
        tree_len = max(tree_len, tree.values[i] + 1);
    }
    s.tree.hist[lane] = tree_count;
    s.tree.depth[lane] = 0;
    s.tree.bits[lane] = 0;
    __syncwarp();
    warp_write_prefix_histograms(s.tree, tree_len, detail::NUM_TREE_CONTEXTS, w, s.data.code_length,
                                 s.data.tree, lane);
    for (std::size_t i{0}; i < tree.count; ++i) {
        w.write(s.tree.depth[tree.values[i]], s.tree.bits[tree.values[i]]);
    }
}

// One warp per (Dc group, blob): even blocks build the group's pre blob (DC
// prefix code), odd blocks its mid blob (AcMetadata prefix code). Mirrors
// detail::serialize_dc_pre / serialize_dc_mid.
__global__ void dc_group_blobs_kernel(std::size_t bw, std::size_t bh, std::size_t xdg,
                                      const std::uint32_t* dc_hist, const std::uint32_t* am_hist,
                                      const std::size_t* fbc, std::uint8_t* dc_depth,
                                      std::uint16_t* dc_bits, std::uint8_t* am_depth,
                                      std::uint16_t* am_bits, std::uint8_t* blob_pre,
                                      std::uint32_t* blob_pre_off, std::uint32_t* blob_pre_bits,
                                      std::uint8_t* blob_mid, std::uint32_t* blob_mid_off,
                                      std::uint32_t* blob_mid_bits) {
    __shared__ DcBlobScratch s;
    const unsigned int lane{threadIdx.x};
    const std::size_t g{blockIdx.x / 2};
    const bool mid{(blockIdx.x & 1) != 0};

    load_histogram((mid ? am_hist : dc_hist) + g * HISTOGRAM_STRIDE, s.data.code, lane);
    for (std::size_t i{lane}; i < DC_BLOB_STRIDE / 4; i += CODE_THREADS) {
        s.words[i] = 0;
    }
    __syncwarp();

    SharedBitWriter w{s.words, 0, lane == 0};
    if (mid) {
        const std::size_t gx{g % xdg};
        const std::size_t gy{g / xdg};
        const std::size_t dgw{DC_GROUP_BLOCKS < bw - gx * DC_GROUP_BLOCKS
                                  ? DC_GROUP_BLOCKS
                                  : bw - gx * DC_GROUP_BLOCKS};
        const std::size_t dgh{DC_GROUP_BLOCKS < bh - gy * DC_GROUP_BLOCKS
                                  ? DC_GROUP_BLOCKS
                                  : bh - gy * DC_GROUP_BLOCKS};
        // AcMetadata `count` field.
        bitstream::write_bits(w, detail::ceil_log2_nonzero(dgw * dgh),
                              static_cast<std::uint32_t>(fbc[g] - 1));
        warp_write_modular_tree(w, detail::acmeta_tree(), s, lane);
        const HistogramSummary data{summarize_histogram(s.data.code.hist, HISTOGRAM_STRIDE, lane)};
        detail::write_acmeta_histograms_head(w, data.alphabet);
        warp_write_prefix_code(s.data.code, data.alphabet, w, s.data.code_length, s.data.tree,
                               lane);
    } else {
        bitstream::write_bits(w, 2, 0);  // DecodeVarDCTDC extra_precision = 0
        warp_write_modular_tree(w, detail::single_leaf_tree(DC_PREDICTOR_GRADIENT), s, lane);
        const HistogramSummary data{summarize_histogram(s.data.code.hist, HISTOGRAM_STRIDE, lane)};
        warp_write_prefix_histograms(s.data.code, data.alphabet, 1, w, s.data.code_length,
                                     s.data.tree, lane);
    }
    assert(w.pos <= DC_BLOB_STRIDE * 8);
    __syncwarp();

    store_code(s.data.code, (mid ? am_depth : dc_depth) + g * HISTOGRAM_STRIDE,
               (mid ? am_bits : dc_bits) + g * HISTOGRAM_STRIDE, lane);
    auto* blob_words{
        reinterpret_cast<unsigned int*>((mid ? blob_mid : blob_pre) + g * DC_BLOB_STRIDE)};
    for (std::uint32_t i{lane}; i < (w.pos + 31) / 32; i += CODE_THREADS) {
        blob_words[i] = s.words[i];
    }
    if (lane == 0) {
        (mid ? blob_mid_off : blob_pre_off)[g] = static_cast<std::uint32_t>(g * DC_BLOB_STRIDE);
        (mid ? blob_mid_bits : blob_pre_bits)[g] = w.pos;
    }
}

}  // namespace

bool build_dc_group_blobs_device(std::size_t width, std::size_t height,
                                 const std::uint32_t* d_dc_hist, const std::uint32_t* d_am_hist,
                                 const std::size_t* d_fbc, std::uint8_t* d_dc_depth,
                                 std::uint16_t* d_dc_bits, std::uint8_t* d_am_depth,
                                 std::uint16_t* d_am_bits, std::uint8_t* d_blob_pre,
                                 std::uint32_t* d_blob_pre_off, std::uint32_t* d_blob_pre_bits,
                                 std::uint8_t* d_blob_mid, std::uint32_t* d_blob_mid_off,
                                 std::uint32_t* d_blob_mid_bits, cudaStream_t stream) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{ceil_div(bw, DC_GROUP_BLOCKS)};
    const std::size_t ydg{ceil_div(bh, DC_GROUP_BLOCKS)};
    const std::size_t num_groups{xdg * ydg};

    dc_group_blobs_kernel<<<static_cast<unsigned int>(2 * num_groups), CODE_THREADS, 0, stream>>>(
        bw, bh, xdg, d_dc_hist, d_am_hist, d_fbc, d_dc_depth, d_dc_bits, d_am_depth, d_am_bits,
        d_blob_pre, d_blob_pre_off, d_blob_pre_bits, d_blob_mid, d_blob_mid_off, d_blob_mid_bits);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace cujpegxl
