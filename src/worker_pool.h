// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_WORKER_POOL_H_
#define CUJPEGXL_SRC_WORKER_POOL_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace cujpegxl {

// Persistent threads that run one task at a time on all of them. The threads
// sleep between tasks.
class WorkerPool {
public:
    // `num_threads` (at least 1) counts the thread calling run().
    explicit WorkerPool(std::size_t num_threads);
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;
    ~WorkerPool();

    std::size_t size() const { return threads_.size() + 1; }

    // Runs task(t) for every t in [0, size()), task(0) on the calling thread, and
    // returns when all have returned.
    void run(const std::function<void(std::size_t)>& task);

private:
    void loop(std::size_t t);

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    const std::function<void(std::size_t)>* task_{nullptr};
    std::uint64_t generation_{0};
    std::size_t running_{0};
    bool stop_{false};
};

// Half the hardware threads (the physical cores of SMT hosts), at least 1.
std::size_t default_worker_threads();

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_WORKER_POOL_H_
