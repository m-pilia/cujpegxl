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

// The device encoder of the C ABI, a pipeline of frames: submit() enqueues a
// frame's device work and returns, a host thread then entropy-codes and
// assembles it, and receive() returns the results in submission order. Up to
// `pipeline_depth` frames are in flight (submitted and not yet received), each
// in its own slot of device scratch, pinned staging buffers, streams and
// events, so that the device work of a frame runs while the host codes the
// frames before it. submit() and receive() must not be called concurrently.
class FrameEncoder {
public:
    // Frames are encoded on `device_ordinal`. `worker_threads` (at least 1)
    // host threads code one frame at a time; `pipeline_depth` is at least 1.
    FrameEncoder(std::int32_t device_ordinal, std::size_t worker_threads,
                 std::size_t pipeline_depth);
    FrameEncoder(const FrameEncoder&) = delete;
    FrameEncoder& operator=(const FrameEncoder&) = delete;
    // Waits for the frames in flight to complete.
    ~FrameEncoder();

    std::size_t pipeline_depth() const;
    std::size_t in_flight() const;

    // Starts encoding a device-resident NV12 image into the `.jxl` file written
    // to the host buffer `out_file` of `out_capacity` bytes. Requires
    // in_flight() < pipeline_depth(). The image must stay unchanged and
    // `out_file` and `stats` valid until the frame is received. False, with
    // the frame not in flight, on a CUDA error or for a single-AC-group frame.
    //
    // `luma`/`chroma` are device addresses (see nv12_to_xyb). The chroma texture
    // requires a 32-byte-aligned row pitch; if `chroma_pitch` is not aligned the
    // plane is first copied into an aligned buffer. `distance` drives
    // quantization; `qp` is the quantizer state written into the codestream.
    // width and height must be multiples of 8 (single-AC-group frames use the
    // combined-section layout and are not handled here).
    //
    // The device runs the transform selection, chroma-from-luma estimate,
    // residual+quantize, codes the DcGroups and tokenizes the AC groups; the
    // host builds the AC entropy code (AcGlobal) from the device histograms,
    // codes each AC group's tokens on the worker threads as they arrive, and
    // assembles the headers, TOC and sections in `out_file`. The output does
    // not depend on the number of worker threads, the pipeline depth or the
    // other frames in flight.
    //
    // When `stats` is non-null it receives the "frontend", "entropy" and
    // "assembly" stage timings; the front end is then waited for in submit().
    [[nodiscard]] bool submit(const std::uint8_t* luma, std::size_t luma_pitch,
                              const std::uint8_t* chroma, std::size_t chroma_pitch,
                              std::size_t width, std::size_t height, float distance,
                              const bitstream::QuantParams& qp, std::uint8_t* out_file,
                              std::size_t out_capacity, std::vector<StageTiming>* stats = nullptr);

    // Waits for the oldest frame in flight and returns its result. Requires
    // in_flight() > 0. The file size is known before the first write to
    // `out_file`, so BUFFER_TOO_SMALL leaves the buffer untouched. FAILED on a
    // CUDA error or if a coded DC section exceeds its capacity.
    [[nodiscard]] EncodeResult receive();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRAME_ENCODER_H_
