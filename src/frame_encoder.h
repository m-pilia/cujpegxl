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

// Encodes a device-resident quantized DCT8 coefficient buffer into a complete
// ISOBMFF `.jxl` file, returned in `out_file` (host bytes).
//
// `q_device` is the quantizer output: three tightly packed planes (X, Y, B) of
// `width/8 * height/8` blocks in raster order, 64 coefficients each, DC in slot
// 0. width/height are multiples of 8 and must span more than one AC group (the
// M1 ladder; single 256x256 frames use the combined-section layout and are not
// yet handled here). `quant_field` is the per-block quant integer buffer (device;
// width/8 * height/8, block raster order) written as the AcMetadata quant field.
//
// The AC groups and DcGroups are entropy-coded on the device; the codestream
// headers, FrameHeader, DcGlobal, AcGlobal, TOC and ISOBMFF boxes are assembled
// on the host from the device histograms and section sizes (all O(1)/O(groups)/
// O(alphabet)); the body is gathered into one device buffer and copied out with
// a single bulk D2H. Returns false on a CUDA error, if a coded section exceeds
// its worst-case buffer, or if the frame has a single AC group.
// When `stats` is non-null it receives the "entropy" and "assembly" stage
// timings for the budget model; passing null (the default) emits no records.
bool encode_frame(const std::int32_t* q_device, std::size_t width, std::size_t height,
                  const bitstream::QuantParams& qp, const std::int32_t* quant_field,
                  std::vector<std::uint8_t>& out_file, std::vector<StageTiming>* stats = nullptr);

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

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRAME_ENCODER_H_
