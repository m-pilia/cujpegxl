# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote


def resolve_image_path(root, path):
    pending = list(PurePosixPath(path).parts)
    resolved = []
    links = 0
    while pending:
        part = pending.pop(0)
        if part == "/":
            resolved = []
        elif part == "..":
            if not resolved:
                raise ValueError(f"Path escapes image root: {path}")
            resolved.pop()
        elif part != ".":
            candidate = root.joinpath(*resolved, part)
            if candidate.is_symlink():
                links += 1
                if links > 40:
                    raise ValueError(f"Symlink loop in image: {path}")
                pending = list(PurePosixPath(os.readlink(candidate)).parts) + pending
            else:
                if not candidate.exists():
                    raise ValueError(f"Missing image file: {path}")
                if pending and not candidate.is_dir():
                    raise ValueError(
                        f"Not an image directory: {candidate.relative_to(root)}"
                    )
                resolved.append(part)
    result = root.joinpath(*resolved)
    if not result.exists():
        raise ValueError(f"Missing image file: {path}")
    return result


def installed_packages(root, names):
    records = {}
    for paragraph in (root / "var/lib/dpkg/status").read_text().split("\n\n"):
        fields = dict(
            line.split(": ", 1)
            for line in paragraph.splitlines()
            if ": " in line and not line.startswith(" ")
        )
        if fields.get("Package") in names:
            name = fields["Package"]
            if fields.get("Status") != "install ok installed":
                raise ValueError(f"Required package is not installed: {name}")
            if fields.get("Architecture") != "arm64":
                raise ValueError(f"Required package is not ARM64: {name}")
            source = re.fullmatch(
                r"([a-z0-9][a-z0-9+.-]*)(?: \(([^()\s]+)\))?",
                fields.get("Source", name),
            )
            if source is None:
                raise ValueError(f"Invalid source package: {name}")
            records[name] = {
                "version": fields["Version"],
                "architecture": fields["Architecture"],
                "source": source[1],
                "source_version": source[2] or fields["Version"],
            }
    missing = sorted(set(names) - records.keys())
    if missing:
        raise ValueError(f"Missing development packages: {', '.join(missing)}")
    return records


def materialize_packages(root, output, packages):
    output.mkdir()
    for name, record in packages.items():
        info = root / "var/lib/dpkg/info"
        listing = info / f"{name}:{record['architecture']}.list"
        if not listing.exists():
            listing = info / f"{name}.list"
        if not listing.exists():
            raise ValueError(f"Missing package file list: {name}")
        for entry in listing.read_text().splitlines():
            path = PurePosixPath(entry)
            if ".." in path.parts or not path.is_absolute():
                raise ValueError(f"Invalid package path: {entry}")
            if not any(
                path.is_relative_to(prefix)
                for prefix in ("/usr/include", "/usr/lib", "/lib", "/lib64")
            ):
                continue
            source = resolve_image_path(root, entry)
            if source.is_dir():
                continue
            if not source.is_file():
                raise ValueError(f"Not a regular sysroot file: {entry}")
            destination = output / path.relative_to("/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Resolve links in the image's namespace, never against the worker's /lib.
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
        copyright_path = resolve_image_path(root, f"/usr/share/doc/{name}/copyright")
        documentation = output / "usr/share/doc" / name
        documentation.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(copyright_path, documentation / "copyright")

    licenses = output / "usr/share/common-licenses"
    licenses.mkdir(parents=True)
    for entry in sorted(
        resolve_image_path(root, "/usr/share/common-licenses").iterdir()
    ):
        source = resolve_image_path(root, f"/usr/share/common-licenses/{entry.name}")
        if not source.is_file():
            raise ValueError(f"Not a regular license file: {entry.name}")
        shutil.copyfile(source, licenses / entry.name)

    sources = [
        "# Sysroot package sources\n",
        "The links below identify the exact Ubuntu source packages for the included",
        "ARM64 binaries. Each source page provides the original source archives,",
        "Ubuntu packaging and patches, and a `.dsc` file with checksums.\n",
        "Package copyright notices are under `usr/share/doc/<package>/copyright`.",
        "The image's standard license texts are under `usr/share/common-licenses/`.\n",
        "| Binary package | Binary version | Source package | Source version |",
        "| --- | --- | --- | --- |",
    ]
    for name, record in sorted(packages.items()):
        source = record["source"]
        version = record["source_version"]
        url = f"https://launchpad.net/ubuntu/+source/{quote(source, safe='')}/{quote(version, safe='')}"
        sources.append(
            f"| {name} | {record['version']} | [{source}]({url}) | {version} |"
        )
    (output / "SOURCES.md").write_text("\n".join(sources) + "\n")


def validate_sysroot(root):
    required = [
        "usr/include/stdio.h",
        "usr/include/stdint.h",
        "usr/include/features.h",
        "usr/include/pthread.h",
        "usr/include/dlfcn.h",
        "usr/include/linux/types.h",
        "usr/include/c++/11/vector",
        "usr/include/aarch64-linux-gnu/c++/11/bits/c++config.h",
        "lib/ld-linux-aarch64.so.1",
    ]
    required += [
        f"usr/lib/aarch64-linux-gnu/{name}"
        for name in (
            "crt1.o",
            "Scrt1.o",
            "crti.o",
            "crtn.o",
            "libc.so",
            "libc_nonshared.a",
            "libm.so",
        )
    ]
    required += [
        f"usr/lib/gcc/aarch64-linux-gnu/11/{name}"
        for name in (
            "crtbegin.o",
            "crtbeginS.o",
            "crtend.o",
            "crtendS.o",
            "libgcc.a",
            "libgcc_s.so",
            "libstdc++.so",
            "libstdc++.a",
            "libsupc++.a",
        )
    ]
    missing = [path for path in required if not (root / path).is_file()]
    if missing:
        raise ValueError("Incomplete sysroot; missing files:\n" + "\n".join(missing))

    library_dirs = [
        root / "lib/aarch64-linux-gnu",
        root / "usr/lib/aarch64-linux-gnu",
        root / "usr/lib/gcc/aarch64-linux-gnu/11",
    ]
    versions = {}
    elf_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as source:
            header = source.read(20)
            if header.startswith(b"\x7fELF"):
                if (
                    len(header) != 20
                    or header[4:6] != b"\x02\x01"
                    or struct.unpack_from("<H", header, 18)[0] != 183
                ):
                    raise ValueError(
                        f"Not an AArch64 ELF64 file: {path.relative_to(root)}"
                    )
                elf_count += 1
                if path.name in ("libc.so.6", "libstdc++.so.6", "libgcc_s.so.1"):
                    data = header + source.read()
                    maxima = {}
                    for family in ("GLIBC", "GLIBCXX", "CXXABI"):
                        matches = re.findall(
                            rb"(?<=\x00)("
                            + family.encode()
                            + rb"_[0-9]+(?:\.[0-9]+)+)(?=\x00)",
                            data,
                        )
                        if matches:
                            maxima[family] = max(
                                matches,
                                key=lambda value: tuple(
                                    map(int, value.split(b"_")[1].split(b"."))
                                ),
                            ).decode()
                    versions[str(path.relative_to(root))] = maxima
            elif path.suffix == ".so":
                text = (header + source.read()).decode()
                text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
                for token in re.findall(r"[^\s()]+", text):
                    if token.startswith("/"):
                        candidates = [root / token.lstrip("/")]
                    elif token.startswith("-l"):
                        candidates = [
                            directory / f"lib{token[2:]}.a"
                            for directory in library_dirs
                        ]
                    elif re.search(r"\.(?:so(?:\.[0-9]+)*|a)$", token):
                        candidates = [path.parent / token] + [
                            directory / token for directory in library_dirs
                        ]
                    else:
                        continue
                    if not any(
                        candidate.resolve().is_relative_to(root) and candidate.is_file()
                        for candidate in candidates
                    ):
                        raise ValueError(
                            f"Unresolved linker script input {token} in {path.relative_to(root)}"
                        )
    if not versions or not elf_count:
        raise ValueError("Sysroot has no validated runtime libraries")
    for name in ("libc.so.6", "libstdc++.so.6", "libgcc_s.so.1"):
        if not any(Path(path).name == name for path in versions):
            raise ValueError(f"Missing runtime library: {name}")
    return {"elf_files": elf_count, "runtime_symbol_version_maxima": versions}


def prepare_sysroot(image, umoci, packages, output):
    index = json.loads((image / "index.json").read_text())
    if len(index["manifests"]) != 1:
        raise ValueError("Sysroot image must contain exactly one platform manifest")
    descriptor = index["manifests"][0]
    algorithm, digest = descriptor["digest"].split(":", 1)
    manifest = json.loads((image / "blobs" / algorithm / digest).read_text())
    algorithm, digest = manifest["config"]["digest"].split(":", 1)
    config = json.loads((image / "blobs" / algorithm / digest).read_text())
    if (config["os"], config["architecture"]) != ("linux", "arm64"):
        raise ValueError("Sysroot image must target Linux ARM64")
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        work = Path(temporary)
        layout = work / "image"
        layout.mkdir()
        descriptor.setdefault("annotations", {})[
            "org.opencontainers.image.ref.name"
        ] = "sysroot"
        (layout / "index.json").write_text(json.dumps(index))
        (layout / "oci-layout").write_text((image / "oci-layout").read_text())
        (layout / "blobs").symlink_to(image / "blobs", target_is_directory=True)
        subprocess.run(
            [
                str(umoci),
                "unpack",
                "--rootless",
                "--image",
                f"{layout}:sysroot",
                str(work / "bundle"),
            ],
            check=True,
        )
        root = work / "bundle/rootfs"
        records = installed_packages(root, packages)
        materialize_packages(root, output / "sysroot", records)
        report = validate_sysroot(output / "sysroot")
        report.update(
            {
                "image_manifest": descriptor["digest"],
                "architecture": "aarch64-linux-gnu",
                "packages": records,
                "os_release": resolve_image_path(root, "/etc/os-release").read_text(),
            }
        )
        (output / "abi.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--umoci", type=Path, required=True)
    parser.add_argument("packages", nargs="+")
    args = parser.parse_args()
    try:
        prepare_sysroot(args.image, args.umoci, args.packages, Path.cwd())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Sysroot validation failed: {error}\n")


if __name__ == "__main__":
    main()
