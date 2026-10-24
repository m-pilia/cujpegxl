// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include <jxl/encode.h>
#include <jxl/thread_parallel_runner.h>

#include "tools/benchmark/rgb_bench_driver.h"

namespace {

class LibjxlEncoder {
public:
    explicit LibjxlEncoder(const RgbBenchArgs& args)
        : width_{args.width},
          height_{args.height},
          distance_{args.distance},
          effort_{args.effort},
          encoder_{JxlEncoderCreate(nullptr), JxlEncoderDestroy},
          runner_{JxlThreadParallelRunnerCreate(nullptr,
                                                JxlThreadParallelRunnerDefaultNumWorkerThreads()),
                  JxlThreadParallelRunnerDestroy} {
        if (!encoder_ || !runner_) {
            std::fprintf(stderr, "could not create libjxl encoder or thread runner\n");
            std::exit(1);
        }
    }

    std::size_t encode(const std::vector<std::uint8_t>& rgb) {
        JxlEncoderReset(encoder_.get());
        if (JxlEncoderSetParallelRunner(encoder_.get(), JxlThreadParallelRunner, runner_.get()) !=
            JXL_ENC_SUCCESS) {
            fail("could not set thread runner");
        }
        JxlBasicInfo info{};
        JxlEncoderInitBasicInfo(&info);
        info.xsize = width_;
        info.ysize = height_;
        info.bits_per_sample = 8;
        info.num_color_channels = 3;
        info.uses_original_profile = JXL_FALSE;
        if (JxlEncoderSetBasicInfo(encoder_.get(), &info) != JXL_ENC_SUCCESS) {
            fail("could not set image information");
        }
        JxlColorEncoding color{};
        JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
        if (JxlEncoderSetColorEncoding(encoder_.get(), &color) != JXL_ENC_SUCCESS) {
            fail("could not set color encoding");
        }
        JxlEncoderFrameSettings* settings{JxlEncoderFrameSettingsCreate(encoder_.get(), nullptr)};
        if (settings == nullptr ||
            JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort_) !=
                JXL_ENC_SUCCESS ||
            JxlEncoderSetFrameDistance(settings, distance_) != JXL_ENC_SUCCESS) {
            fail("could not set frame settings");
        }
        const JxlPixelFormat pixel_format{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
        if (JxlEncoderAddImageFrame(settings, &pixel_format, rgb.data(), rgb.size()) !=
            JXL_ENC_SUCCESS) {
            fail("could not add image frame");
        }
        JxlEncoderCloseInput(encoder_.get());

        output_.resize(1 << 16);
        std::size_t used{0};
        for (;;) {
            std::uint8_t* next{output_.data() + used};
            std::size_t available{output_.size() - used};
            const JxlEncoderStatus status{
                JxlEncoderProcessOutput(encoder_.get(), &next, &available)};
            used = output_.size() - available;
            if (status == JXL_ENC_SUCCESS) {
                return used;
            }
            if (status != JXL_ENC_NEED_MORE_OUTPUT) {
                fail("encoding failed");
            }
            output_.resize(output_.size() * 2);
        }
    }

private:
    [[noreturn]] static void fail(const char* message) {
        std::fprintf(stderr, "libjxl: %s\n", message);
        std::exit(1);
    }

    std::uint32_t width_;
    std::uint32_t height_;
    float distance_;
    std::int32_t effort_;
    std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> encoder_;
    std::unique_ptr<void, decltype(&JxlThreadParallelRunnerDestroy)> runner_;
    std::vector<std::uint8_t> output_;
};

}  // namespace

int main(int argc, char** argv) {
    const RgbBenchArgs args{parse_rgb_bench_args(argc, argv, false)};
    LibjxlEncoder encoder{args};
    return run_rgb_bench(args, "libjxl", false,
                         [&](const auto& rgb) { return encoder.encode(rgb); });
}
