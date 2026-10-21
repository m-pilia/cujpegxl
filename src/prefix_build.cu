// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "prefix_build.h"

#include <cassert>

#include <cuda_runtime.h>

#include "ac_context.h"
#include "dc_predict.h"
#include "src/bitstream/bit_sink.h"
#include "src/bitstream/frame_assembly.h"
#include "src/bitstream/serialize_impl.h"
#include "vardct_layout.h"
#include "warp_scan.cuh"

namespace cujpegxl {
namespace {

using bitstream::CountingBitWriter;
using bitstream::HISTOGRAM_STRIDE;
using bitstream::HybridUintConfig;
namespace detail = bitstream::detail;

constexpr unsigned int FULL_MASK{~0u};

// Each prefix code (AC cluster, DC or AcMetadata group histogram) is built and
// described by one warp, as a single-warp block.
constexpr unsigned int CODE_THREADS{32};

constexpr unsigned int AC_SERIALIZE_THREADS{256};

// Shared-memory staging for the AcGlobal section, sized well above the real
// section (~1.4 KB) and the structural worst case.
constexpr std::size_t AC_SECTION_SHMEM = 24576;

// Byte slot reserved per cluster for its serialized prefix-code description. A
// 256-symbol code description is a few hundred bits; this bounds it with margin.
constexpr std::size_t CLUSTER_BLOB_STRIDE = 512;

// Code lengths of a length-limited (<= 15 bits) code are below this.
constexpr std::uint32_t DEPTH_VALUES{16};

constexpr std::size_t MAX_TREE_NODES{2 * detail::MAX_ALPHABET + 4};

static_assert(AC_NUM_CLUSTERS == 32, "ac_serialize_kernel maps one lane per cluster");

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

// Writes the `nbits` LSB-first bits of the byte blob `src` at bit `dst_pos`, one
// 32-bit chunk per participating thread (`thread` of `num_threads`).
__device__ void copy_bits(unsigned int* words, std::uint32_t dst_pos, const std::uint8_t* src,
                          std::uint32_t nbits, unsigned int thread, unsigned int num_threads) {
    const std::uint32_t nbytes{(nbits + 7) / 8};
    for (std::uint32_t chunk{thread}; chunk * 32 < nbits; chunk += num_threads) {
        std::uint32_t value{0};
        for (std::uint32_t b{0}; b < 4; ++b) {
            const std::uint32_t byte{chunk * 4 + b};
            if (byte < nbytes) {
                value |= static_cast<std::uint32_t>(src[byte]) << (8 * b);
            }
        }
        const std::uint32_t n{min(32u, nbits - chunk * 32)};
        if (n < 32) {
            value &= (1u << n) - 1u;
        }
        SharedBitWriter{words, dst_pos + chunk * 32, true}.write(n, value);
    }
}

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

// Warp-cooperative detail::write_prefix_histograms.
__device__ void warp_write_prefix_histograms(HistogramCode& code, std::uint32_t length,
                                             std::size_t num_contexts, SharedBitWriter& w,
                                             CodeLengthCode& cl, TreeScratch& t,
                                             unsigned int lane) {
    detail::write_prefix_histogram_head(w, length, num_contexts, HybridUintConfig{});
    if (length > 1) {
        const HistogramSummary s{summarize_histogram(code.hist, length, lane)};
        warp_build_and_store_huffman_code(code, length, s, w, cl, t, lane);
    }
}

struct DcBlobScratch {
    CodeScratch data;
    HistogramCode tree;
    unsigned int words[DC_BLOB_STRIDE / 4];
};

// Warp-cooperative detail::write_modular_header over the staged data histogram.
__device__ void warp_write_modular_header(SharedBitWriter& w, int predictor, DcBlobScratch& s,
                                          unsigned int lane) {
    w.write(1, 0);  // use_global_tree = false
    w.write(1, 1);  // weighted::Header all_default = true
    w.write(2, 0);  // num_transforms = 0

    const std::uint32_t tree_tokens[detail::NUM_TREE_CONTEXTS - 1]{
        0, static_cast<std::uint32_t>(predictor), 0, 0, 0};
    std::uint32_t tree_count{0};
    std::uint32_t tree_len{1};
    for (std::uint32_t token : tree_tokens) {
        tree_count += token == lane;
        tree_len = max(tree_len, token + 1);
    }
    s.tree.hist[lane] = tree_count;
    s.tree.depth[lane] = 0;
    s.tree.bits[lane] = 0;
    __syncwarp();
    warp_write_prefix_histograms(s.tree, tree_len, detail::NUM_TREE_CONTEXTS, w, s.data.code_length,
                                 s.data.tree, lane);
    for (std::uint32_t token : tree_tokens) {
        w.write(s.tree.depth[token], s.tree.bits[token]);
    }

    const HistogramSummary data{summarize_histogram(s.data.code.hist, HISTOGRAM_STRIDE, lane)};
    warp_write_prefix_histograms(s.data.code, data.alphabet, 1, w, s.data.code_length, s.data.tree,
                                 lane);
}

// One warp per AC cluster: builds the cluster's prefix code and serializes its
// code description into its own blob slot.
__global__ void ac_build_codes_kernel(const std::uint32_t* ac_hist, std::uint8_t* depth,
                                      std::uint16_t* bits, std::uint8_t* cluster_blob,
                                      std::uint32_t* cluster_alpha, std::uint32_t* cluster_nbits) {
    __shared__ CodeScratch s;
    __shared__ unsigned int s_words[CLUSTER_BLOB_STRIDE / 4];
    const unsigned int lane{threadIdx.x};
    const std::size_t c{blockIdx.x};

    load_histogram(ac_hist + c * HISTOGRAM_STRIDE, s.code, lane);
    for (std::size_t i{lane}; i < CLUSTER_BLOB_STRIDE / 4; i += CODE_THREADS) {
        s_words[i] = 0;
    }
    __syncwarp();

    const HistogramSummary summary{summarize_histogram(s.code.hist, HISTOGRAM_STRIDE, lane)};
    SharedBitWriter w{s_words, 0, lane == 0};
    if (summary.alphabet > 1) {
        warp_build_and_store_huffman_code(s.code, summary.alphabet, summary, w, s.code_length,
                                          s.tree, lane);
    }
    assert(w.pos <= CLUSTER_BLOB_STRIDE * 8);
    __syncwarp();

    store_code(s.code, depth + c * HISTOGRAM_STRIDE, bits + c * HISTOGRAM_STRIDE, lane);
    auto* blob_words{reinterpret_cast<unsigned int*>(cluster_blob + c * CLUSTER_BLOB_STRIDE)};
    for (std::uint32_t i{lane}; i < (w.pos + 31) / 32; i += CODE_THREADS) {
        blob_words[i] = s_words[i];
    }
    if (lane == 0) {
        cluster_alpha[c] = summary.alphabet;
        cluster_nbits[c] = w.pos;
    }
}

// Serializes the AcGlobal section: the fixed head, the precomputed context map,
// the per-cluster uint configs and alphabet sizes, and the cluster code blobs,
// each placed at its scanned bit offset and copied in parallel.
__global__ void ac_serialize_kernel(std::size_t num_ac_groups, const std::uint8_t* ctxmap_blob,
                                    std::uint32_t ctxmap_nbits, const std::uint32_t* cluster_alpha,
                                    const std::uint8_t* cluster_blob,
                                    const std::uint32_t* cluster_nbits, std::uint8_t* section,
                                    std::uint32_t* section_len) {
    __shared__ unsigned int s_words[AC_SECTION_SHMEM / 4];
    __shared__ std::uint32_t s_alpha_pos[AC_NUM_CLUSTERS];
    __shared__ std::uint32_t s_blob_pos[AC_NUM_CLUSTERS];
    __shared__ std::uint32_t s_total_bits;
    const unsigned int tid{threadIdx.x};
    const unsigned int lane{tid & 31u};
    const unsigned int warp{tid >> 5};

    CountingBitWriter head{};
    detail::serialize_ac_global_head(head, num_ac_groups);
    head.write(1, 0);  // lz77.enabled = false
    CountingBitWriter config{};
    detail::encode_uint_config(config, HybridUintConfig{}, detail::PREFIX_MAX_BITS);
    const auto ctxmap_pos{static_cast<std::uint32_t>(head.bits_written())};
    const std::uint32_t use_prefix_pos{ctxmap_pos + ctxmap_nbits};
    const auto config_bits{static_cast<std::uint32_t>(config.bits_written())};
    const std::uint32_t alpha_pos{use_prefix_pos + 1 + AC_NUM_CLUSTERS * config_bits};

    if (warp == 0) {
        CountingBitWriter alpha_field{};
        detail::store_var_len_uint16(alpha_field, cluster_alpha[lane] - 1);
        const auto alpha_bits{static_cast<std::uint32_t>(alpha_field.bits_written())};
        const std::uint32_t blob_bits{cluster_alpha[lane] > 1 ? cluster_nbits[lane] : 0};
        const std::uint32_t alpha_end{warp_inclusive_sum(alpha_bits, lane)};
        const std::uint32_t blob_end{warp_inclusive_sum(blob_bits, lane)};
        const std::uint32_t blobs_pos{alpha_pos + __shfl_sync(FULL_MASK, alpha_end, 31)};
        s_alpha_pos[lane] = alpha_pos + alpha_end - alpha_bits;
        s_blob_pos[lane] = blobs_pos + blob_end - blob_bits;
        if (lane == 31) {
            s_total_bits = blobs_pos + blob_end;
        }
    }
    __syncthreads();
    const std::uint32_t total_bits{s_total_bits};
    assert(total_bits <= AC_SECTION_SHMEM * 8);
    for (std::uint32_t i{tid}; i < (total_bits + 31) / 32; i += blockDim.x) {
        s_words[i] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        SharedBitWriter w{s_words, 0, true};
        detail::serialize_ac_global_head(w, num_ac_groups);
        w.write(1, 0);                                               // lz77.enabled = false
        SharedBitWriter{s_words, use_prefix_pos, true}.write(1, 1);  // use_prefix_code
    }
    if (tid < AC_NUM_CLUSTERS) {
        SharedBitWriter config_w{s_words, use_prefix_pos + 1 + tid * config_bits, true};
        detail::encode_uint_config(config_w, HybridUintConfig{}, detail::PREFIX_MAX_BITS);
        SharedBitWriter alpha_w{s_words, s_alpha_pos[tid], true};
        detail::store_var_len_uint16(alpha_w, cluster_alpha[tid] - 1);
    }
    copy_bits(s_words, ctxmap_pos, ctxmap_blob, ctxmap_nbits, tid, blockDim.x);
    for (unsigned int c{warp}; c < AC_NUM_CLUSTERS; c += blockDim.x / 32) {
        if (cluster_alpha[c] > 1) {
            copy_bits(s_words, s_blob_pos[c], cluster_blob + c * CLUSTER_BLOB_STRIDE,
                      cluster_nbits[c], lane, 32);
        }
    }
    __syncthreads();

    const std::uint32_t num_bytes{(total_bits + 7) / 8};
    const auto* bytes{reinterpret_cast<const std::uint8_t*>(s_words)};
    for (std::uint32_t i{tid}; i < num_bytes; i += blockDim.x) {
        section[i] = bytes[i];
    }
    if (tid == 0) {
        *section_len = num_bytes;
    }
}

// One warp per (Dc group, blob): even blocks build the group's pre blob (DC
// prefix code), odd blocks its mid blob (AcMetadata prefix code).
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
        warp_write_modular_header(w, DC_PREDICTOR_ZERO, s, lane);
    } else {
        bitstream::write_bits(w, 2, 0);  // DecodeVarDCTDC extra_precision = 0
        warp_write_modular_header(w, DC_PREDICTOR_GRADIENT, s, lane);
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

bool build_ac_global_device(const std::uint32_t* d_ac_hist, std::size_t num_ac_groups,
                            const std::uint8_t* d_ctxmap_blob, std::size_t ctxmap_nbits,
                            std::uint8_t* d_depth, std::uint16_t* d_bits, std::uint8_t* d_section,
                            std::size_t section_cap, std::uint32_t* d_section_len) {
    assert(section_cap >= AC_SECTION_SHMEM);
    (void)section_cap;
    std::uint8_t* d_cluster_blob{nullptr};
    std::uint32_t* d_cluster_meta{nullptr};
    if (cudaMallocAsync(&d_cluster_blob, AC_NUM_CLUSTERS * CLUSTER_BLOB_STRIDE, 0) != cudaSuccess ||
        cudaMallocAsync(&d_cluster_meta, 2 * AC_NUM_CLUSTERS * sizeof(std::uint32_t), 0) !=
            cudaSuccess) {
        return false;
    }
    std::uint32_t* d_cluster_alpha{d_cluster_meta};
    std::uint32_t* d_cluster_nbits{d_cluster_meta + AC_NUM_CLUSTERS};
    ac_build_codes_kernel<<<AC_NUM_CLUSTERS, CODE_THREADS>>>(
        d_ac_hist, d_depth, d_bits, d_cluster_blob, d_cluster_alpha, d_cluster_nbits);
    ac_serialize_kernel<<<1, AC_SERIALIZE_THREADS>>>(
        num_ac_groups, d_ctxmap_blob, static_cast<std::uint32_t>(ctxmap_nbits), d_cluster_alpha,
        d_cluster_blob, d_cluster_nbits, d_section, d_section_len);
    const bool ok{cudaGetLastError() == cudaSuccess};
    cudaFreeAsync(d_cluster_blob, 0);
    cudaFreeAsync(d_cluster_meta, 0);
    return ok;
}

bool build_dc_group_blobs_device(std::size_t width, std::size_t height,
                                 const std::uint32_t* d_dc_hist, const std::uint32_t* d_am_hist,
                                 const std::size_t* d_fbc, std::uint8_t* d_dc_depth,
                                 std::uint16_t* d_dc_bits, std::uint8_t* d_am_depth,
                                 std::uint16_t* d_am_bits, std::uint8_t* d_blob_pre,
                                 std::uint32_t* d_blob_pre_off, std::uint32_t* d_blob_pre_bits,
                                 std::uint8_t* d_blob_mid, std::uint32_t* d_blob_mid_off,
                                 std::uint32_t* d_blob_mid_bits) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{ceil_div(bw, DC_GROUP_BLOCKS)};
    const std::size_t ydg{ceil_div(bh, DC_GROUP_BLOCKS)};
    const std::size_t num_groups{xdg * ydg};

    dc_group_blobs_kernel<<<static_cast<unsigned int>(2 * num_groups), CODE_THREADS>>>(
        bw, bh, xdg, d_dc_hist, d_am_hist, d_fbc, d_dc_depth, d_dc_bits, d_am_depth, d_am_bits,
        d_blob_pre, d_blob_pre_off, d_blob_pre_bits, d_blob_mid, d_blob_mid_off, d_blob_mid_bits);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace cujpegxl
