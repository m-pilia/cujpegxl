// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRAME_ENCODER_H_
#define CUJPEGXL_SRC_FRAME_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "src/bitstream/frame_assembly.h"

namespace cujpegxl {

// Per-stage instrumentation record. `gpu_us`/`cpu_us` are measured wall time of
// the device and host portions of a stage; because the device sub-calls are
// synchronous, blocking wall time attributes cleanly to each. `bytes_moved` is
// the analytic DRAM traffic of the stage.
struct StageTiming {
    const char* name{""};
    std::size_t bytes_moved{0};
    double gpu_us{0.0};
    double cpu_us{0.0};
};

enum class EncodeStatus : std::uint8_t { OK, BUFFER_TOO_SMALL, FAILED };

struct EncodeResult {
    EncodeStatus status{EncodeStatus::FAILED};
    // The bytes written on OK; the capacity the file needs on BUFFER_TOO_SMALL.
    std::size_t file_size{0};
};

// Capacities of the coded sections: the device DcGroups buffer (an encode fails
// when the DcGroups outgrow it) and bounds of the AcGroups and AcGlobal, so
// that together they bound the coded body of a frame.
struct SectionCapacities {
    std::size_t ac_groups{0};
    std::size_t dc_groups{0};
    std::size_t ac_global{0};
};

SectionCapacities section_capacities(std::size_t width, std::size_t height);

// Upper bound of the `.jxl` file size of a width x height frame: with an output
// buffer of this capacity, an encode never reports BUFFER_TOO_SMALL.
std::size_t max_encoded_size(std::size_t width, std::size_t height);

// Maps a Butteraugli `distance` to the serialized Quantizer state written into
// the codestream. Placeholder linear mapping, not yet calibrated against cjxl's
// Butteraugli semantics.
bitstream::QuantParams quant_params_for_distance(float distance);

// The device encoder of the C ABI. It owns the resources reused across frames:
// the host worker threads that entropy-code the AC groups, the pinned staging
// buffers and the streams and events. One frame is encoded at a time.
class FrameEncoder {
public:
    // `worker_threads` (at least 1) counts the thread calling encode_nv12.
    explicit FrameEncoder(std::size_t worker_threads);
    FrameEncoder(const FrameEncoder&) = delete;
    FrameEncoder& operator=(const FrameEncoder&) = delete;
    ~FrameEncoder();

    // Selects `device_ordinal`, runs the transform selection, chroma-from-luma
    // estimate, residual+quantize and entropy coding over a device-resident NV12
    // image and writes the `.jxl` file into the host buffer `out_file` of
    // `out_capacity` bytes.
    //
    // `luma`/`chroma` are device addresses (see nv12_to_xyb). The chroma texture
    // requires a 32-byte-aligned row pitch; if `chroma_pitch` is not aligned the
    // plane is copied into an internally allocated aligned buffer. `distance`
    // drives quantization; `qp` is the quantizer state written into the
    // codestream. width and height must be multiples of 8 and span more than one
    // AC group (single 256x256 frames use the combined-section layout and are not
    // handled here).
    //
    // The device codes the DcGroups and tokenizes the AC groups; the host builds
    // the AC entropy code (AcGlobal) from the device histograms, codes each AC
    // group's tokens on the worker threads as they arrive, and assembles the
    // headers, TOC and sections in `out_file`. The file size is known before the
    // first write to `out_file`, so BUFFER_TOO_SMALL leaves the buffer untouched.
    // FAILED on a CUDA error, if a coded DC section exceeds its capacity, or for
    // a single-AC-group frame. The output does not depend on the number of
    // worker threads.
    //
    // When `stats` is non-null it receives the "frontend", "entropy" and
    // "assembly" stage timings.
    [[nodiscard]] EncodeResult encode_nv12(const std::uint8_t* luma, std::size_t luma_pitch,
                                           const std::uint8_t* chroma, std::size_t chroma_pitch,
                                           std::size_t width, std::size_t height,
                                           std::int32_t device_ordinal, float distance,
                                           const bitstream::QuantParams& qp, std::uint8_t* out_file,
                                           std::size_t out_capacity,
                                           std::vector<StageTiming>* stats = nullptr);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRAME_ENCODER_H_
