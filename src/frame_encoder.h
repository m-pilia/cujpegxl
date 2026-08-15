// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRAME_ENCODER_H_
#define CUJPEGXL_SRC_FRAME_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <memory>
#include <vector>

#include "src/bitstream/frame_assembly.h"

namespace cujpegxl {

// Per-stage instrumentation record consumed by the budget model
// (tools/budget/budget_model.py). `gpu_us`/`cpu_us` are measured wall time of
// the device and host portions of a stage; because the device sub-calls are
// synchronous, blocking wall time attributes cleanly to each. `bytes_moved` is
// the analytic DRAM traffic of the stage.
struct StageTiming {
    const char* name{""};
    std::size_t bytes_moved{0};
    double gpu_us{0.0};
    double cpu_us{0.0};
};

enum class AcClusteringMode : std::uint8_t {
    DATA_DRIVEN = 0,
    FIXED = 1,
};

enum class EncoderPipeline : std::uint8_t {
    DCT8 = 0,
    MIXED = 1,
};

struct EncoderConfig {
    std::int32_t device_ordinal{0};
    std::size_t max_width{3840};
    std::size_t max_height{2160};
    // Depth 1 minimizes latency; deeper queues can increase throughput by
    // overlapping CPU clustering with GPU work from another frame.
    std::size_t pipeline_depth{2};
    EncoderPipeline pipeline{EncoderPipeline::MIXED};
};

struct EncoderInput {
    // Device pointers must remain valid until the returned future becomes ready.
    const std::uint8_t* luma{nullptr};
    std::size_t luma_pitch{0};
    const std::uint8_t* chroma{nullptr};
    std::size_t chroma_pitch{0};
    std::size_t width{0};
    std::size_t height{0};
    float distance{1.0f};
    bitstream::QuantParams quant_params{};
    std::uint64_t sequence{0};
    bool collect_stats{false};
};

struct EncodedFrame {
    std::uint64_t sequence{0};
    std::vector<std::uint8_t> bytes{};
    std::vector<StageTiming> stats{};
};

class EncodedFrameFuture {
public:
    EncodedFrameFuture() = default;

    bool valid() const;
    bool ready() const;
    bool wait_for(std::chrono::nanoseconds timeout) const;
    bool get(EncodedFrame& output);

private:
    struct State;
    explicit EncodedFrameFuture(std::shared_ptr<State> state);
    std::shared_ptr<State> state_{};
    friend class EncoderSession;
};

class EncoderSession {
public:
    static std::unique_ptr<EncoderSession> create(const EncoderConfig& config);
    ~EncoderSession();

    EncoderSession(const EncoderSession&) = delete;
    EncoderSession& operator=(const EncoderSession&) = delete;

    // `try_encode` reports backpressure without blocking. `encode` waits only
    // for capacity; completion is observed through the returned future.
    bool try_encode(const EncoderInput& input, EncodedFrameFuture& future);
    bool encode(const EncoderInput& input, EncodedFrameFuture& future);
    void flush();

private:
    struct Impl;
    explicit EncoderSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Encodes device-resident quantized DCT8 coefficients into a complete ISOBMFF
// `.jxl` file, returned in `out_file` (host bytes).
//
// The coefficients are split by bandwidth: `dc_device` holds the int32 DC (three
// channel-major planes X, Y, B; one DC per block; width/8 * height/8 each), and
// `ac_device` the packed int16 AC (three channel-major planes X, Y, B;
// AC_COEFFS_PER_BLOCK coefficients per block with the DC slot elided, so
// libjxl-raster coefficient index k in [1, 63] lands at slot k-1). width/height
// are multiples of 8 and must span more than one AC group (the M1 ladder; single
// 256x256 frames use the combined-section layout and are not yet handled here).
// `quant_field` is the per-block quant integer buffer (device; width/8 * height/8,
// block raster order) written as the AcMetadata quant field.
//
// The AC groups and DcGroups are entropy-coded on the device; the codestream
// headers, FrameHeader, DcGlobal, AcGlobal, TOC and ISOBMFF boxes are assembled
// on the host from the device histograms and section sizes (all O(1)/O(groups)/
// O(alphabet)); the body is gathered into one device buffer and copied out with
// a single bulk D2H. Returns false on a CUDA error, if a coded section exceeds
// its worst-case buffer, or if the frame has a single AC group.
// When `stats` is non-null it receives the "entropy" and "assembly" stage
// timings for the budget model; passing null (the default) emits no records.
bool encode_frame(const std::int16_t* ac_device, const std::int32_t* dc_device, std::size_t width,
                  std::size_t height, const bitstream::QuantParams& qp,
                  const std::int32_t* quant_field, std::vector<std::uint8_t>& out_file,
                  std::vector<StageTiming>* stats = nullptr,
                  AcClusteringMode clustering = AcClusteringMode::DATA_DRIVEN);

// Maps a Butteraugli `distance` to the serialized Quantizer state written into
// the codestream. Placeholder linear mapping; calibrating it against cjxl's
// Butteraugli semantics is the W7 conformance item.
bitstream::QuantParams quant_params_for_distance(float distance);

// Full device encode path for the C ABI: selects `device_ordinal`, runs
// nv12_to_xyb -> forward_dct8 -> quantize_dct8 -> encode_frame over a
// device-resident NV12 image and returns the `.jxl` file bytes in `out_file`.
//
// `luma`/`chroma` are device addresses (see nv12_to_xyb). The chroma texture
// requires a 32-byte-aligned row pitch; if `chroma_pitch` is not aligned the
// plane is copied into an internally allocated aligned buffer. `distance` drives
// quantize_dct8; `qp` is the quantizer state written into the codestream. width
// and height must be multiples of 8 and span more than one AC group. Returns
// false on a CUDA error or an unsupported single-AC-group frame.
// When `stats` is non-null it receives the "frontend", "entropy" and "assembly"
// stage timings for the budget model.
bool encode_nv12(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                 std::size_t chroma_pitch, std::size_t width, std::size_t height,
                 std::int32_t device_ordinal, float distance, const bitstream::QuantParams& qp,
                 std::vector<std::uint8_t>& out_file, std::vector<StageTiming>* stats = nullptr);

// M3 encode: the mixed-block {8,16,32} + chroma-from-luma path. Assembles a
// codestream from device-resident quantized coefficients in the covered-block
// layout (`ac_device`: three int16 planes of (width/8 * height/8) * 64 slots;
// `dc_device`: three int32 planes, one DC per 8x8 block), the per-8x8 transform
// signal `acs`, the per-64x64-tile CfL maps `ytox_map`/`ytob_map`, and the
// per-block `quant_field`. Uses the mixed-block device entropy path. Same
// constraints as encode_frame (multi-AC-group ladder only).
bool encode_frame_m3(const std::int16_t* ac_device, const std::int32_t* dc_device,
                     const std::int8_t* acs, const std::int8_t* ytox_map,
                     const std::int8_t* ytob_map, std::size_t width, std::size_t height,
                     const bitstream::QuantParams& qp, const std::int32_t* quant_field,
                     std::vector<std::uint8_t>& out_file,
                     std::vector<StageTiming>* stats = nullptr,
                     AcClusteringMode clustering = AcClusteringMode::DATA_DRIVEN);

// M3 full device encode path (K1 select+transform, K2 CfL estimate, K3
// residual+quantize, mixed-block entropy + assembly). Arguments as encode_nv12.
bool encode_nv12_m3(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                    std::size_t chroma_pitch, std::size_t width, std::size_t height,
                    std::int32_t device_ordinal, float distance, const bitstream::QuantParams& qp,
                    std::vector<std::uint8_t>& out_file, std::vector<StageTiming>* stats = nullptr);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRAME_ENCODER_H_
