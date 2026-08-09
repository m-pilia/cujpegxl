# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
project(cujpegxl_cmake_consumer C)

find_package(cujpegxl REQUIRED)

add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE cujpegxl::cujpegxl)
