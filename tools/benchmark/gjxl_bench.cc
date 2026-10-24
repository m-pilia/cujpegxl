// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include <gjxl/gjxl.h>

#include "tools/benchmark/rgb_bench_driver.h"

namespace {

[[noreturn]] void fail(const char* operation) {
    std::fprintf(stderr, "gjxl %s failed: %s\n", operation, gjxl_get_last_error());
    std::exit(1);
}

class GjxlEncoder {
public:
    explicit GjxlEncoder(const RgbBenchArgs& args)
        : width_{args.width}, height_{args.height}, context_{nullptr, gjxl_context_destroy} {
        if (args.device != 0) {
            std::fprintf(
                stderr,
                "gjxl uses CUDA device 0; set CUDA_VISIBLE_DEVICES to select another GPU\n");
            std::exit(1);
        }
        GJXLContextOptions context_options{};
        if (gjxl_context_options_init(&context_options, sizeof(context_options)) != GJXL_OK) {
            fail("context options initialization");
        }
        context_options.backend = GJXL_BACKEND_CUDA;
        GJXLContext* context{nullptr};
        if (gjxl_context_create(&context_options, &context) != GJXL_OK) {
            fail("context creation");
        }
        context_.reset(context);
        if (gjxl_encoder_options_init(&encoder_options_, sizeof(encoder_options_)) != GJXL_OK) {
            fail("encoder options initialization");
        }
        encoder_options_.distance = args.distance;
        encoder_options_.effort = args.effort;
    }

    std::size_t encode(const std::vector<std::uint8_t>& rgb) {
        const GJXLImageView image{
            .struct_size = sizeof(GJXLImageView),
            .width = width_,
            .height = height_,
            .pixel_format = GJXL_PIXEL_FORMAT_RGB8_SRGB,
            .pixels = rgb.data(),
            .pixels_size = rgb.size(),
            .row_stride_bytes = static_cast<std::size_t>(width_) * 3,
        };
        GJXLBuffer output{};
        if (gjxl_encode(context_.get(), &image, &encoder_options_, &output) != GJXL_OK) {
            fail("encode");
        }
        const std::size_t encoded_size{output.size};
        gjxl_buffer_free(&output);
        return encoded_size;
    }

private:
    std::uint32_t width_;
    std::uint32_t height_;
    std::unique_ptr<GJXLContext, decltype(&gjxl_context_destroy)> context_;
    GJXLEncoderOptions encoder_options_{};
};

}  // namespace

int main(int argc, char** argv) {
    const RgbBenchArgs args{parse_rgb_bench_args(argc, argv, true)};
    GjxlEncoder encoder{args};
    return run_rgb_bench(args, "gjxl", true, [&](const auto& rgb) { return encoder.encode(rgb); });
}
