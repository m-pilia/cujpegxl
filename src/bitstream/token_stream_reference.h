// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_TOKEN_STREAM_REFERENCE_H_
#define CUJPEGXL_SRC_BITSTREAM_TOKEN_STREAM_REFERENCE_H_

#include <cstdint>
#include <vector>

#include "token_stream.h"

namespace cujpegxl::bitstream {

// Straightforward coder of one token stream, structured as libjxl's WriteTokens:
// for rANS a backward pass with exact division records each token's
// renormalization chunk, then the final state, the chunks and the raw bits are
// written forward. The reference for code_token_streams.
std::vector<std::uint8_t> code_token_stream_reference(const TokenCode& code,
                                                      const TokenStream& stream);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_TOKEN_STREAM_REFERENCE_H_
