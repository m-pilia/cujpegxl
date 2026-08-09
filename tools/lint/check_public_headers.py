# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Assert that public headers expose no CUDA types or includes.

Non-CUDA consumers must be able to compile and link against the shipped headers
without a CUDA toolkit, so the public C ABI must not leak any CUDA identifiers.
"""

import re
import sys

# Targets real CUDA leakage (includes, runtime/driver identifiers, device
# qualifiers) rather than the word "CUDA" in prose, so explanatory comments that
# state the header is CUDA-free do not trip the check.
_FORBIDDEN = re.compile(
    r"#\s*include\s*[<\"][^>\"]*cuda"  # cuda_runtime.h / cuda.h includes
    r"|\bcuda[A-Z_]\w*"  # runtime API: cudaMalloc, cuda_runtime, cudaError_t
    r"|\bCU[a-z]\w*"  # driver types: CUdeviceptr, CUstream, CUcontext
    r"|__device__|__global__|__host__|__constant__|__shared__"
    r"|\bnvcc\b"
)


def scan(path: str) -> list[str]:
    offending = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if _FORBIDDEN.search(line):
                offending.append(f"{path}:{lineno}: {line.strip()}")
    return offending


def main(argv: list[str]) -> int:
    paths = argv[1:]
    if not paths:
        print("no public headers provided", file=sys.stderr)
        return 2

    offending: list[str] = []
    for path in paths:
        offending.extend(scan(path))

    if offending:
        print("Public headers must be CUDA-free. Offending lines:")
        print("\n".join(offending))
        return 1

    print(f"OK: {len(paths)} public header(s) are CUDA-free.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
