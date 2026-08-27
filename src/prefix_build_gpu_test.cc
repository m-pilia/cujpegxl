// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// De-risks the on-device prefix-code build (prefix_build.cu) by asserting it
// reproduces the host bitstream reference (build_ac_global / build_dc_group_blobs)
// byte-for-byte over histograms spanning every Huffman path: empty, single
// symbol, the <=4-symbol simple form, the complex code-length-code form, and the
// full 256-symbol alphabet. Both sides run the same serialize_impl.h core, so a
// mismatch would flag a device-specific defect (bounded-buffer sizing, the stable
// sort, the explicit-stack depth walk, or the strided blob layout).

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "prefix_build.h"
#include "src/ac_context.h"
#include "src/bitstream/frame_assembly.h"
#include "src/entropy.h"

namespace cujpegxl {
namespace {

using bitstream::AcGlobalResult;
using bitstream::DcGroupBlobs;
using bitstream::HISTOGRAM_STRIDE;

// Deterministic 32-bit LCG so every case is byte-reproducible run to run.
struct Lcg {
    std::uint64_t state;
    std::uint32_t next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(state >> 32);
    }
};

// Fills one HISTOGRAM_STRIDE histogram with a pattern selected by `pattern`,
// covering each distinct prefix-code path.
void fill_histogram(std::uint32_t* h, int pattern, Lcg& rng) {
    for (std::size_t i{0}; i < HISTOGRAM_STRIDE; ++i) {
        h[i] = 0;
    }
    switch (pattern % 6) {
        case 0:  // empty (alphabet 1, no code)
            break;
        case 1:  // single symbol
            h[rng.next() % HISTOGRAM_STRIDE] = 1 + rng.next() % 1000;
            break;
        case 2:  // two symbols (simple form)
            h[3] = 10;
            h[200] = 4;
            break;
        case 3:  // four symbols (simple form boundary)
            h[0] = 7;
            h[1] = 3;
            h[50] = 9;
            h[255] = 1;
            break;
        case 4: {  // sparse many symbols (complex code-length form)
            const int n{5 + static_cast<int>(rng.next() % 40)};
            for (int k{0}; k < n; ++k) {
                h[rng.next() % HISTOGRAM_STRIDE] += 1 + rng.next() % 500;
            }
            break;
        }
        case 5:  // full alphabet, skewed counts
            for (std::size_t i{0}; i < HISTOGRAM_STRIDE; ++i) {
                h[i] = 1 + rng.next() % (i + 2);
            }
            break;
    }
}

template <typename T>
T* device_upload(const std::vector<T>& host) {
    T* d{nullptr};
    EXPECT_EQ(cudaMalloc(&d, host.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
              cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> device_download(const T* d, std::size_t count) {
    std::vector<T> host(count);
    EXPECT_EQ(cudaMemcpy(host.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return host;
}

// Bytes spanning `nbits` of a byte-aligned writer (both sides zero the unused
// tail of the final byte, so a full-byte compare is exact).
std::size_t byte_span(std::uint32_t nbits) {
    return (nbits + 7) / 8;
}

void check_ac_global(std::size_t width, std::size_t height, Lcg& rng) {
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    std::vector<std::uint32_t> ac_hist(AC_NUM_CLUSTERS * HISTOGRAM_STRIDE);
    for (int c{0}; c < AC_NUM_CLUSTERS; ++c) {
        fill_histogram(ac_hist.data() + c * HISTOGRAM_STRIDE, c, rng);
    }

    const AcGlobalResult host{bitstream::build_ac_global(ac_hist.data(), num_ac)};
    const bitstream::AcContextMapBlob ctxmap{bitstream::build_ac_context_map_blob()};

    std::uint32_t* d_hist{device_upload(ac_hist)};
    std::uint8_t* d_ctxmap{device_upload(ctxmap.bytes)};
    std::uint8_t* d_depth{nullptr};
    std::uint16_t* d_bits{nullptr};
    std::uint8_t* d_section{nullptr};
    std::uint32_t* d_section_len{nullptr};
    const std::size_t section_cap{1u << 16};
    ASSERT_EQ(cudaMalloc(&d_depth, AC_NUM_CLUSTERS * HISTOGRAM_STRIDE), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bits, AC_NUM_CLUSTERS * HISTOGRAM_STRIDE * sizeof(std::uint16_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_section, section_cap), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_section_len, sizeof(std::uint32_t)), cudaSuccess);

    ASSERT_TRUE(build_ac_global_device(d_hist, num_ac, d_ctxmap, ctxmap.nbits, d_depth, d_bits,
                                       d_section, section_cap, d_section_len));

    const auto dev_depth{device_download(d_depth, AC_NUM_CLUSTERS * HISTOGRAM_STRIDE)};
    const auto dev_bits{device_download(d_bits, AC_NUM_CLUSTERS * HISTOGRAM_STRIDE)};
    const auto dev_len{device_download(d_section_len, 1)};
    const auto dev_section{device_download(d_section, dev_len[0])};

    EXPECT_EQ(dev_len[0], host.section.size());
    EXPECT_EQ(dev_section, host.section);
    EXPECT_EQ(dev_depth, host.depth);
    EXPECT_EQ(dev_bits, host.bits);

    cudaFree(d_hist);
    cudaFree(d_ctxmap);
    cudaFree(d_depth);
    cudaFree(d_bits);
    cudaFree(d_section);
    cudaFree(d_section_len);
}

void check_dc_group_blobs(std::size_t width, std::size_t height, Lcg& rng) {
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    std::vector<std::uint32_t> dc_hist(num_dc * HISTOGRAM_STRIDE);
    std::vector<std::uint32_t> am_hist(num_dc * HISTOGRAM_STRIDE);
    std::vector<std::size_t> fbc(num_dc);
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{(bw + 255) / 256};
    for (std::size_t g{0}; g < num_dc; ++g) {
        fill_histogram(dc_hist.data() + g * HISTOGRAM_STRIDE, static_cast<int>(g) + 1, rng);
        fill_histogram(am_hist.data() + g * HISTOGRAM_STRIDE, static_cast<int>(g) + 3, rng);
        const std::size_t gx{g % xdg};
        const std::size_t gy{g / xdg};
        const std::size_t dgw{std::min<std::size_t>(256, bw - gx * 256)};
        const std::size_t dgh{std::min<std::size_t>(256, bh - gy * 256)};
        const std::size_t total{dgw * dgh};
        fbc[g] = 1 + (rng.next() % total);
    }

    const DcGroupBlobs host{bitstream::build_dc_group_blobs(width, height, dc_hist.data(),
                                                            am_hist.data(), fbc.data())};

    std::uint32_t* d_dc_hist{device_upload(dc_hist)};
    std::uint32_t* d_am_hist{device_upload(am_hist)};
    std::size_t* d_fbc{device_upload(fbc)};
    std::uint8_t* d_dc_depth{nullptr};
    std::uint16_t* d_dc_bits{nullptr};
    std::uint8_t* d_am_depth{nullptr};
    std::uint16_t* d_am_bits{nullptr};
    std::uint8_t* d_pre{nullptr};
    std::uint32_t* d_pre_off{nullptr};
    std::uint32_t* d_pre_bits{nullptr};
    std::uint8_t* d_mid{nullptr};
    std::uint32_t* d_mid_off{nullptr};
    std::uint32_t* d_mid_bits{nullptr};
    ASSERT_EQ(cudaMalloc(&d_dc_depth, num_dc * HISTOGRAM_STRIDE), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dc_bits, num_dc * HISTOGRAM_STRIDE * sizeof(std::uint16_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_am_depth, num_dc * HISTOGRAM_STRIDE), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_am_bits, num_dc * HISTOGRAM_STRIDE * sizeof(std::uint16_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pre, num_dc * DC_BLOB_STRIDE), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pre_off, num_dc * sizeof(std::uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pre_bits, num_dc * sizeof(std::uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_mid, num_dc * DC_BLOB_STRIDE), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_mid_off, num_dc * sizeof(std::uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_mid_bits, num_dc * sizeof(std::uint32_t)), cudaSuccess);

    ASSERT_TRUE(build_dc_group_blobs_device(width, height, d_dc_hist, d_am_hist, d_fbc, d_dc_depth,
                                            d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off,
                                            d_pre_bits, d_mid, d_mid_off, d_mid_bits));

    const auto dev_dc_depth{device_download(d_dc_depth, num_dc * HISTOGRAM_STRIDE)};
    const auto dev_dc_bits{device_download(d_dc_bits, num_dc * HISTOGRAM_STRIDE)};
    const auto dev_am_depth{device_download(d_am_depth, num_dc * HISTOGRAM_STRIDE)};
    const auto dev_am_bits{device_download(d_am_bits, num_dc * HISTOGRAM_STRIDE)};
    const auto dev_pre{device_download(d_pre, num_dc * DC_BLOB_STRIDE)};
    const auto dev_pre_bits{device_download(d_pre_bits, num_dc)};
    const auto dev_mid{device_download(d_mid, num_dc * DC_BLOB_STRIDE)};
    const auto dev_mid_bits{device_download(d_mid_bits, num_dc)};

    EXPECT_EQ(dev_dc_depth, host.dc_depth);
    EXPECT_EQ(dev_dc_bits, host.dc_bits);
    EXPECT_EQ(dev_am_depth, host.acmeta_depth);
    EXPECT_EQ(dev_am_bits, host.acmeta_bits);

    for (std::size_t g{0}; g < num_dc; ++g) {
        EXPECT_EQ(dev_pre_bits[g], host.blob_pre_bits[g]) << "pre bits group " << g;
        EXPECT_EQ(dev_mid_bits[g], host.blob_mid_bits[g]) << "mid bits group " << g;
        for (std::size_t b{0}; b < byte_span(host.blob_pre_bits[g]); ++b) {
            EXPECT_EQ(dev_pre[g * DC_BLOB_STRIDE + b], host.blob_pre[host.blob_pre_off[g] + b])
                << "pre byte " << b << " group " << g;
        }
        for (std::size_t b{0}; b < byte_span(host.blob_mid_bits[g]); ++b) {
            EXPECT_EQ(dev_mid[g * DC_BLOB_STRIDE + b], host.blob_mid[host.blob_mid_off[g] + b])
                << "mid byte " << b << " group " << g;
        }
    }

    cudaFree(d_dc_hist);
    cudaFree(d_am_hist);
    cudaFree(d_fbc);
    cudaFree(d_dc_depth);
    cudaFree(d_dc_bits);
    cudaFree(d_am_depth);
    cudaFree(d_am_bits);
    cudaFree(d_pre);
    cudaFree(d_pre_off);
    cudaFree(d_pre_bits);
    cudaFree(d_mid);
    cudaFree(d_mid_off);
    cudaFree(d_mid_bits);
}

TEST(PrefixBuildGpu, AcGlobalMatchesHost) {
    Lcg rng{0x1234567u};
    for (int iter{0}; iter < 8; ++iter) {
        check_ac_global(1920, 1080, rng);
        check_ac_global(3840, 2160, rng);
    }
}

TEST(PrefixBuildGpu, DcGroupBlobsMatchHost) {
    Lcg rng{0x89abcdefu};
    for (int iter{0}; iter < 8; ++iter) {
        check_dc_group_blobs(1920, 1080, rng);
        check_dc_group_blobs(3840, 2160, rng);
    }
}

}  // namespace
}  // namespace cujpegxl
