# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path

from prepare_sysroot import installed_packages, materialize_packages, resolve_image_path


class ImagePathsTest(unittest.TestCase):
    def test_absolute_links_and_merged_usr_resolve_inside_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "usr/lib").mkdir(parents=True)
            (root / "usr/lib/runtime.so").write_bytes(b"target runtime")
            (root / "lib").symlink_to("usr/lib")
            (root / "usr/lib/link.so").symlink_to("/lib/runtime.so")

            resolved = resolve_image_path(root, "/lib/link.so")

            self.assertEqual(resolved, root / "usr/lib/runtime.so")
            self.assertEqual(resolved.read_bytes(), b"target runtime")

    def test_rejects_links_escaping_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "escape").symlink_to("../outside")

            with self.assertRaisesRegex(ValueError, "escapes image root"):
                resolve_image_path(root, "/escape")

    def test_rejects_dangling_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "missing").symlink_to("/absent")

            with self.assertRaisesRegex(ValueError, "Missing image file"):
                resolve_image_path(root, "/missing")

    def test_rejects_link_cycles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "first").symlink_to("second")
            (root / "second").symlink_to("first")

            with self.assertRaisesRegex(ValueError, "Symlink loop"):
                resolve_image_path(root, "/first")

    def test_rejects_traversal_through_regular_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file").write_text("not a directory")
            (root / "link").symlink_to("file/../file")

            with self.assertRaisesRegex(ValueError, "Not an image directory"):
                resolve_image_path(root, "/link")


class PackagesTest(unittest.TestCase):
    def test_materializes_only_package_owned_development_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = workspace / "image"
            files = {
                "var/lib/dpkg/status": (
                    "Package: libc6-dev\nStatus: install ok installed\n"
                    "Architecture: arm64\nVersion: 2.35+b1\nSource: glibc (2.35)\n"
                ),
                "var/lib/dpkg/info/libc6-dev:arm64.list": (
                    "/.\n/usr/include\n/usr/include/example.h\n"
                    "/usr/lib/link.so\n/usr/bin/target-tool\n"
                ),
                "usr/include/example.h": "target header",
                "usr/lib/runtime.so": "target library",
                "usr/lib/unrelated.so": "unrelated library",
                "usr/bin/target-tool": "target executable",
                "usr/share/doc/libc6-dev/copyright": "license",
                "usr/share/common-licenses/GPL-3": "GPL version 3 license text",
            }
            for name, contents in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
            (root / "usr/lib/link.so").symlink_to("/usr/lib/runtime.so")
            (root / "usr/share/common-licenses/GPL").symlink_to(
                "/usr/share/common-licenses/GPL-3"
            )

            packages = installed_packages(root, ["libc6-dev"])
            materialize_packages(root, workspace / "sysroot", packages)

            self.assertEqual(packages["libc6-dev"]["version"], "2.35+b1")
            self.assertEqual(
                (workspace / "sysroot/usr/include/example.h").read_text(),
                "target header",
            )
            self.assertEqual(
                (workspace / "sysroot/usr/lib/link.so").read_text(),
                "target library",
            )
            self.assertFalse((workspace / "sysroot/usr/lib/link.so").is_symlink())
            self.assertFalse((workspace / "sysroot/usr/lib/unrelated.so").exists())
            self.assertFalse((workspace / "sysroot/usr/bin").exists())
            self.assertEqual(
                (workspace / "sysroot/usr/share/doc/libc6-dev/copyright").read_text(),
                "license",
            )
            for name in ("GPL", "GPL-3"):
                path = workspace / "sysroot/usr/share/common-licenses" / name
                self.assertEqual(path.read_text(), "GPL version 3 license text")
                self.assertFalse(path.is_symlink())
            self.assertIn(
                "| libc6-dev | 2.35+b1 | "
                "[glibc](https://launchpad.net/ubuntu/+source/glibc/2.35) | 2.35 |",
                (workspace / "sysroot/SOURCES.md").read_text(),
            )

    def test_identifies_source_package_and_version(self):
        for source_field, expected_source, expected_version in (
            ("", "libc6", "2.35"),
            ("Source: glibc\n", "glibc", "2.35"),
            ("Source: glibc (2.34)\n", "glibc", "2.34"),
        ):
            with (
                self.subTest(source_field=source_field),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                (root / "var/lib/dpkg").mkdir(parents=True)
                (root / "var/lib/dpkg/status").write_text(
                    "Package: libc6\nStatus: install ok installed\n"
                    "Architecture: arm64\nVersion: 2.35\n" + source_field
                )

                record = installed_packages(root, ["libc6"])["libc6"]

                self.assertEqual(record["source"], expected_source)
                self.assertEqual(record["source_version"], expected_version)

    def test_rejects_malformed_source_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "var/lib/dpkg").mkdir(parents=True)
            (root / "var/lib/dpkg/status").write_text(
                "Package: libc6\nStatus: install ok installed\n"
                "Architecture: arm64\nVersion: 2.35\nSource: glibc (\n"
            )

            with self.assertRaisesRegex(ValueError, "Invalid source package: libc6"):
                installed_packages(root, ["libc6"])

    def test_rejects_missing_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "var/lib/dpkg").mkdir(parents=True)
            (root / "var/lib/dpkg/status").write_text("")

            with self.assertRaisesRegex(
                ValueError, "Missing development packages: libc6-dev"
            ):
                installed_packages(root, ["libc6-dev"])

    def test_rejects_wrong_package_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "var/lib/dpkg").mkdir(parents=True)
            (root / "var/lib/dpkg/status").write_text(
                "Package: libc6-dev\nStatus: install ok installed\n"
                "Architecture: amd64\nVersion: 2.35\n"
            )

            with self.assertRaisesRegex(ValueError, "not ARM64"):
                installed_packages(root, ["libc6-dev"])


if __name__ == "__main__":
    unittest.main()
