// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "worker_pool.h"

#include <algorithm>

namespace cujpegxl {

WorkerPool::WorkerPool(std::size_t num_threads) {
    for (std::size_t t{1}; t < std::max<std::size_t>(num_threads, 1); ++t) {
        threads_.emplace_back([this, t] { loop(t); });
    }
}

WorkerPool::~WorkerPool() {
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        stop_ = true;
    }
    start_.notify_all();
    for (std::thread& t : threads_) {
        t.join();
    }
}

void WorkerPool::run(const std::function<void(std::size_t)>& task) {
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        task_ = &task;
        running_ = threads_.size();
        ++generation_;
    }
    start_.notify_all();
    task(0);
    std::unique_lock<std::mutex> lock{mutex_};
    done_.wait(lock, [this] { return running_ == 0; });
    task_ = nullptr;
}

void WorkerPool::loop(std::size_t t) {
    std::uint64_t seen{0};
    for (;;) {
        const std::function<void(std::size_t)>* task{nullptr};
        {
            std::unique_lock<std::mutex> lock{mutex_};
            start_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) {
                return;
            }
            seen = generation_;
            task = task_;
        }
        (*task)(t);
        bool last{false};
        {
            const std::lock_guard<std::mutex> lock{mutex_};
            --running_;
            last = running_ == 0;
        }
        if (last) {
            done_.notify_one();
        }
    }
}

std::size_t default_worker_threads() {
    return std::max<std::size_t>(1, std::thread::hardware_concurrency() / 2);
}

}  // namespace cujpegxl
