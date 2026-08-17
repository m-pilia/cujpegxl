// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRAME_ENCODER_H_
#define CUJPEGXL_SRC_FRAME_ENCODER_H_

#include <cstddef>
#include <cstdint>
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

// Assembles a complete ISOBMFF `.jxl` file (returned in `out_file`, host bytes)
// from device-resident quantized coefficients in the mixed-block {8,16,32} + CfL
// covered-block layout.
//
// `ac_device` holds three channel-major int16 planes of (width/8 * height/8) * 64
// slots (covered-block layout; a first-block of side N owns its coefficients,
// covered blocks own none); `dc_device` three channel-major int32 planes, one DC
// per 8x8 block. `acs` is the per-8x8 transform signal, `ytox_map`/`ytob_map` the
// per-64x64-tile chroma-from-luma maps, and `quant_field` the per-block quant
// integer buffer (device; width/8 * height/8, block raster order) written as the
// AcMetadata quant field. width/height are multiples of 8 and must span more than
// one AC group (single 256x256 frames use the combined-section layout and are not
// handled here).
//
// The AC groups and DcGroups are entropy-coded on the device; the codestream
// headers, FrameHeader, DcGlobal, AcGlobal, TOC and ISOBMFF boxes are assembled
// on the host from the device histograms and section sizes (all O(1)/O(groups)/
// O(alphabet)); the body is gathered into one device buffer and copied out with
// a single bulk D2H. Returns false on a CUDA error, if a coded section exceeds
// its worst-case buffer, or if the frame has a single AC group.
// When `stats` is non-null it receives the "entropy" and "assembly" stage
// timings for the budget model; passing null (the default) emits no records.
bool encode_frame(const std::int16_t* ac_device, const std::int32_t* dc_device,
                  const std::int8_t* acs, const std::int8_t* ytox_map,
                  const std::int8_t* ytob_map, std::size_t width, std::size_t height,
                  const bitstream::QuantParams& qp, const std::int32_t* quant_field,
                  std::vector<std::uint8_t>& out_file,
                  std::vector<StageTiming>* stats = nullptr);

// Maps a Butteraugli `distance` to the serialized Quantizer state written into
// the codestream. Placeholder linear mapping, not yet calibrated against cjxl's
// Butteraugli semantics.
bitstream::QuantParams quant_params_for_distance(float distance);

// Full device encode path for the C ABI: selects `device_ordinal`, runs the
// transform selection, chroma-from-luma estimate, residual+quantize and
// mixed-block entropy + assembly over a device-resident NV12 image, returning the
// `.jxl` file bytes in `out_file`.
//
// `luma`/`chroma` are device addresses (see nv12_to_xyb). The chroma texture
// requires a 32-byte-aligned row pitch; if `chroma_pitch` is not aligned the
// plane is copied into an internally allocated aligned buffer. `distance` drives
// quantization; `qp` is the quantizer state written into the codestream. width
// and height must be multiples of 8 and span more than one AC group. Returns
// false on a CUDA error or an unsupported single-AC-group frame.
// When `stats` is non-null it receives the "frontend", "entropy" and "assembly"
// stage timings for the budget model.
bool encode_nv12(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                 std::size_t chroma_pitch, std::size_t width, std::size_t height,
                 std::int32_t device_ordinal, float distance, const bitstream::QuantParams& qp,
                 std::vector<std::uint8_t>& out_file, std::vector<StageTiming>* stats = nullptr);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRAME_ENCODER_H_
