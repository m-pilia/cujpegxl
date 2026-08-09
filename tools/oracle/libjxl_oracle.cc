// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Validation oracle: links libjxl's internal encoder to emit reference
// intermediates the GPU encoder is diffed against. Testonly; never part of the
// shipped runtime.
//
// Subcommands read and write planar float32 buffers (3 planes, row-major, no
// padding), so a NumPy caller can round-trip them with numpy.tofile/fromfile.
//
//   xyb <width> <height> <in_srgb_f32> <out_xyb_f32>
//       Input:  sRGB-encoded R,G,B in [0,1]. Output: libjxl opsin XYB.
//
// SDR intensity target is fixed at 255 nits, matching libjxl's default for
// 8-bit sRGB content.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <jxl/cms.h>
#include <jxl/memory_manager.h>

#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_transforms.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/quant_weights.h"

namespace {

constexpr float SDR_INTENSITY_TARGET{255.0f};

void* oracle_alloc(void*, size_t size) {
    return std::malloc(size);
}
void oracle_free(void*, void* address) {
    std::free(address);
}

bool read_planar(const char* path, std::size_t plane_count, std::size_t width, std::size_t height,
                 std::vector<float>* out) {
    const std::size_t count{plane_count * width * height};
    out->resize(count);
    std::FILE* file{std::fopen(path, "rb")};
    if (file == nullptr) {
        std::fprintf(stderr, "oracle: cannot open %s for reading\n", path);
        return false;
    }
    const std::size_t got{std::fread(out->data(), sizeof(float), count, file)};
    std::fclose(file);
    if (got != count) {
        std::fprintf(stderr, "oracle: %s has %zu floats, expected %zu\n", path, got, count);
        return false;
    }
    return true;
}

bool write_planar(const char* path, const std::vector<float>& data) {
    std::FILE* file{std::fopen(path, "wb")};
    if (file == nullptr) {
        std::fprintf(stderr, "oracle: cannot open %s for writing\n", path);
        return false;
    }
    const std::size_t put{std::fwrite(data.data(), sizeof(float), data.size(), file)};
    std::fclose(file);
    if (put != data.size()) {
        std::fprintf(stderr, "oracle: short write to %s\n", path);
        return false;
    }
    return true;
}

int run_xyb(std::size_t width, std::size_t height, const char* in_path, const char* out_path) {
    std::vector<float> rgb{};
    if (!read_planar(in_path, 3, width, height, &rgb)) {
        return 1;
    }

    JxlMemoryManager memory_manager{nullptr, &oracle_alloc, &oracle_free};
    auto image_or = jxl::Image3F::Create(&memory_manager, width, height);
    if (!image_or.ok()) {
        std::fprintf(stderr, "oracle: Image3F::Create failed\n");
        return 1;
    }
    jxl::Image3F image = std::move(image_or).value_();

    const std::size_t plane_stride{width * height};
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::size_t y{0}; y < height; ++y) {
            const float* src{rgb.data() + c * plane_stride + y * width};
            std::memcpy(image.PlaneRow(c, y), src, width * sizeof(float));
        }
    }

    const JxlCmsInterface cms{*JxlGetDefaultCms()};
    if (!jxl::ToXYB(jxl::ColorEncoding::SRGB(), SDR_INTENSITY_TARGET, /*black=*/nullptr,
                    /*pool=*/nullptr, &image, cms, /*linear=*/nullptr)) {
        std::fprintf(stderr, "oracle: ToXYB failed\n");
        return 1;
    }

    std::vector<float> xyb(3 * plane_stride);
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::size_t y{0}; y < height; ++y) {
            float* dst{xyb.data() + c * plane_stride + y * width};
            std::memcpy(dst, image.PlaneRow(c, y), width * sizeof(float));
        }
    }
    return write_planar(out_path, xyb) ? 0 : 1;
}

// Forward 8x8 DCT of every plane, block by block, via libjxl's own transform.
// Output layout per plane: blocks in raster order, 64 coefficients each in
// libjxl's natural (raster) coefficient order.
int run_dct(std::size_t width, std::size_t height, const char* in_path, const char* out_path) {
    if (width % 8 != 0 || height % 8 != 0) {
        std::fprintf(stderr, "oracle: dct requires width and height multiples of 8\n");
        return 2;
    }
    std::vector<float> planes{};
    if (!read_planar(in_path, 3, width, height, &planes)) {
        return 1;
    }

    const std::size_t plane_stride{width * height};
    const std::size_t blocks_x{width / 8};
    const std::size_t blocks_y{height / 8};
    std::vector<float> coeffs(3 * plane_stride);
    // libjxl's transforms expect 64-byte-aligned outputs and a scratch buffer
    // sized for the largest supported transform (3 * lanes * kMaxBlockDim).
    alignas(64) float block[64];
    alignas(64) float scratch[3 * 16 * 256];
    for (std::size_t c{0}; c < 3; ++c) {
        const float* plane{planes.data() + c * plane_stride};
        for (std::size_t by{0}; by < blocks_y; ++by) {
            for (std::size_t bx{0}; bx < blocks_x; ++bx) {
                const float* pixels{plane + (by * 8) * width + bx * 8};
                jxl::TransformFromPixels(static_cast<jxl::AcStrategyType>(0), pixels, width, block,
                                         scratch);
                const std::size_t base{c * plane_stride + (by * blocks_x + bx) * 64};
                for (std::size_t k{0}; k < 64; ++k) {
                    coeffs[base + k] = block[k];
                }
            }
        }
    }
    return write_planar(out_path, coeffs) ? 0 : 1;
}

// Writes libjxl's default DCT8 dequant matrix (3 channels x 64 coefficients, in
// libjxl raster order) as float32, for the encoder to reuse as its quant
// weights.
int run_quantmatrix(const char* out_path) {
    JxlMemoryManager memory_manager{nullptr, &oracle_alloc, &oracle_free};
    jxl::DequantMatrices dm{};
    if (!dm.EnsureComputed(&memory_manager, ~0u)) {
        std::fprintf(stderr, "oracle: EnsureComputed failed\n");
        return 1;
    }
    std::vector<float> weights(3 * 64);
    for (std::size_t c{0}; c < 3; ++c) {
        const float* matrix{dm.Matrix(static_cast<jxl::AcStrategyType>(0), c)};
        for (std::size_t k{0}; k < 64; ++k) {
            weights[c * 64 + k] = matrix[k];
        }
    }
    return write_planar(out_path, weights) ? 0 : 1;
}

std::size_t parse_dim(const char* text) {
    return static_cast<std::size_t>(std::strtoull(text, nullptr, 10));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "xyb") == 0) {
        if (argc != 6) {
            std::fprintf(stderr, "usage: %s xyb <width> <height> <in_srgb_f32> <out_xyb_f32>\n",
                         argv[0]);
            return 2;
        }
        return run_xyb(parse_dim(argv[2]), parse_dim(argv[3]), argv[4], argv[5]);
    }
    if (argc >= 2 && std::strcmp(argv[1], "dct") == 0) {
        if (argc != 6) {
            std::fprintf(stderr, "usage: %s dct <width> <height> <in_xyb_f32> <out_coeff_f32>\n",
                         argv[0]);
            return 2;
        }
        return run_dct(parse_dim(argv[2]), parse_dim(argv[3]), argv[4], argv[5]);
    }
    if (argc == 3 && std::strcmp(argv[1], "quantmatrix") == 0) {
        return run_quantmatrix(argv[2]);
    }

    std::fprintf(stderr,
                 "usage: %s xyb <width> <height> <in_srgb_f32> <out_xyb_f32>\n"
                 "       %s dct <width> <height> <in_xyb_f32> <out_coeff_f32>\n"
                 "       %s quantmatrix <out_weights_f32>\n",
                 argv[0], argv[0], argv[0]);
    return 2;
}
