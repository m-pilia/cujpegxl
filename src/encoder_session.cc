// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_encoder.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "encoder_session_internal.h"

namespace cujpegxl {

struct EncodedFrameFuture::State {
    std::mutex mutex{};
    std::condition_variable ready_cv{};
    EncodedFrame output{};
    bool done{false};
    bool success{false};
    bool consumed{false};
};

struct EncoderSession::Impl {
    struct Job {
        EncoderInput input{};
        std::shared_ptr<EncodedFrameFuture::State> state{};
    };

    explicit Impl(const EncoderConfig& encoder_config) : config{encoder_config} {}

    EncoderConfig config{};
    std::mutex mutex{};
    std::condition_variable job_cv{};
    std::condition_variable capacity_cv{};
    std::condition_variable idle_cv{};
    std::deque<Job> jobs{};
    std::vector<std::thread> workers{};
    std::size_t in_flight{0};
    bool stopping{false};

    bool valid_input(const EncoderInput& input) const {
        return input.luma != nullptr && input.chroma != nullptr && input.width > 0 &&
               input.height > 0 && input.width <= config.max_width &&
               input.height <= config.max_height;
    }

    void run() {
        for (;;) {
            Job job{};
            {
                std::unique_lock<std::mutex> lock{mutex};
                job_cv.wait(lock, [&] { return stopping || !jobs.empty(); });
                if (jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }

            EncodedFrame output{};
            output.sequence = job.input.sequence;
            std::vector<StageTiming>* stats{
                job.input.collect_stats ? &output.stats : nullptr};
            const bool success{
                config.pipeline == EncoderPipeline::MIXED
                    ? encode_nv12_m3_direct(
                          job.input.luma, job.input.luma_pitch,
                          job.input.chroma, job.input.chroma_pitch,
                          job.input.width, job.input.height,
                          config.device_ordinal, job.input.distance,
                          job.input.quant_params, output.bytes, stats)
                    : encode_nv12_direct(
                          job.input.luma, job.input.luma_pitch,
                          job.input.chroma, job.input.chroma_pitch,
                          job.input.width, job.input.height,
                          config.device_ordinal, job.input.distance,
                          job.input.quant_params, output.bytes, stats)};
            {
                std::lock_guard<std::mutex> lock{job.state->mutex};
                job.state->output = std::move(output);
                job.state->success = success;
                job.state->done = true;
            }
            job.state->ready_cv.notify_all();

            {
                std::lock_guard<std::mutex> lock{mutex};
                --in_flight;
                if (in_flight == 0) {
                    idle_cv.notify_all();
                }
            }
            capacity_cv.notify_one();
        }
    }
};

EncodedFrameFuture::EncodedFrameFuture(std::shared_ptr<State> state)
    : state_{std::move(state)} {}

bool EncodedFrameFuture::valid() const { return state_ != nullptr; }

bool EncodedFrameFuture::ready() const {
    if (state_ == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock{state_->mutex};
    return state_->done;
}

bool EncodedFrameFuture::wait_for(std::chrono::nanoseconds timeout) const {
    if (state_ == nullptr) {
        return false;
    }
    std::unique_lock<std::mutex> lock{state_->mutex};
    return state_->ready_cv.wait_for(lock, timeout,
                                     [&] { return state_->done; });
}

bool EncodedFrameFuture::get(EncodedFrame& output) {
    if (state_ == nullptr) {
        return false;
    }
    std::unique_lock<std::mutex> lock{state_->mutex};
    state_->ready_cv.wait(lock, [&] { return state_->done; });
    if (!state_->success || state_->consumed) {
        return false;
    }
    output = std::move(state_->output);
    state_->consumed = true;
    return true;
}

EncoderSession::EncoderSession(std::unique_ptr<Impl> impl)
    : impl_{std::move(impl)} {}

std::unique_ptr<EncoderSession> EncoderSession::create(
    const EncoderConfig& config) {
    if (config.pipeline_depth == 0 || config.max_width == 0 ||
        config.max_height == 0) {
        return nullptr;
    }
    std::unique_ptr<Impl> impl{new Impl{config}};
    impl->workers.reserve(config.pipeline_depth);
    for (std::size_t i{0}; i < config.pipeline_depth; ++i) {
        impl->workers.emplace_back([worker = impl.get()] { worker->run(); });
    }
    return std::unique_ptr<EncoderSession>{
        new EncoderSession{std::move(impl)}};
}

EncoderSession::~EncoderSession() {
    flush();
    {
        std::lock_guard<std::mutex> lock{impl_->mutex};
        impl_->stopping = true;
    }
    impl_->job_cv.notify_all();
    for (std::thread& worker : impl_->workers) {
        worker.join();
    }
}

bool EncoderSession::try_encode(const EncoderInput& input,
                                EncodedFrameFuture& future) {
    if (!impl_->valid_input(input)) {
        return false;
    }
    std::shared_ptr<EncodedFrameFuture::State> state{
        std::make_shared<EncodedFrameFuture::State>()};
    {
        std::lock_guard<std::mutex> lock{impl_->mutex};
        if (impl_->stopping || impl_->in_flight == impl_->config.pipeline_depth) {
            return false;
        }
        impl_->jobs.push_back({input, state});
        ++impl_->in_flight;
    }
    future = EncodedFrameFuture{std::move(state)};
    impl_->job_cv.notify_one();
    return true;
}

bool EncoderSession::encode(const EncoderInput& input,
                            EncodedFrameFuture& future) {
    if (!impl_->valid_input(input)) {
        return false;
    }
    std::shared_ptr<EncodedFrameFuture::State> state{
        std::make_shared<EncodedFrameFuture::State>()};
    {
        std::unique_lock<std::mutex> lock{impl_->mutex};
        impl_->capacity_cv.wait(lock, [&] {
            return impl_->stopping ||
                   impl_->in_flight < impl_->config.pipeline_depth;
        });
        if (impl_->stopping) {
            return false;
        }
        impl_->jobs.push_back({input, state});
        ++impl_->in_flight;
    }
    future = EncodedFrameFuture{std::move(state)};
    impl_->job_cv.notify_one();
    return true;
}

void EncoderSession::flush() {
    std::unique_lock<std::mutex> lock{impl_->mutex};
    impl_->idle_cv.wait(lock, [&] { return impl_->in_flight == 0; });
}

namespace {

struct SynchronousSessions {
    std::int32_t device{-1};
    std::unique_ptr<EncoderSession> dct8{};
    std::unique_ptr<EncoderSession> mixed{};
};

EncoderSession* synchronous_session(std::int32_t device,
                                    EncoderPipeline pipeline) {
    static thread_local SynchronousSessions sessions{};
    if (sessions.device != device) {
        sessions.dct8.reset();
        sessions.mixed.reset();
        sessions.device = device;
    }
    std::unique_ptr<EncoderSession>& selected{
        pipeline == EncoderPipeline::MIXED ? sessions.mixed : sessions.dct8};
    if (selected == nullptr) {
        selected = EncoderSession::create(
            EncoderConfig{.device_ordinal = device,
                          .max_width = 16384,
                          .max_height = 16384,
                          .pipeline_depth = 1,
                          .pipeline = pipeline});
    }
    return selected.get();
}

bool encode_synchronously(
    EncoderPipeline pipeline, const std::uint8_t* luma,
    std::size_t luma_pitch, const std::uint8_t* chroma,
    std::size_t chroma_pitch, std::size_t width, std::size_t height,
    std::int32_t device_ordinal, float distance,
    const bitstream::QuantParams& qp, std::vector<std::uint8_t>& out_file,
    std::vector<StageTiming>* stats) {
    EncoderSession* session{synchronous_session(device_ordinal, pipeline)};
    if (session == nullptr) {
        return false;
    }
    EncoderInput input{.luma = luma,
                       .luma_pitch = luma_pitch,
                       .chroma = chroma,
                       .chroma_pitch = chroma_pitch,
                       .width = width,
                       .height = height,
                       .distance = distance,
                       .quant_params = qp,
                       .sequence = 0,
                       .collect_stats = stats != nullptr};
    EncodedFrameFuture future{};
    EncodedFrame output{};
    if (!session->encode(input, future) || !future.get(output)) {
        return false;
    }
    out_file = std::move(output.bytes);
    if (stats != nullptr) {
        stats->insert(stats->end(), output.stats.begin(), output.stats.end());
    }
    return true;
}

}  // namespace

bool encode_nv12(const std::uint8_t* luma, std::size_t luma_pitch,
                 const std::uint8_t* chroma, std::size_t chroma_pitch,
                 std::size_t width, std::size_t height,
                 std::int32_t device_ordinal, float distance,
                 const bitstream::QuantParams& qp,
                 std::vector<std::uint8_t>& out_file,
                 std::vector<StageTiming>* stats) {
    return encode_synchronously(EncoderPipeline::DCT8, luma, luma_pitch, chroma,
                                chroma_pitch, width, height, device_ordinal,
                                distance, qp, out_file, stats);
}

bool encode_nv12_m3(const std::uint8_t* luma, std::size_t luma_pitch,
                    const std::uint8_t* chroma, std::size_t chroma_pitch,
                    std::size_t width, std::size_t height,
                    std::int32_t device_ordinal, float distance,
                    const bitstream::QuantParams& qp,
                    std::vector<std::uint8_t>& out_file,
                    std::vector<StageTiming>* stats) {
    return encode_synchronously(EncoderPipeline::MIXED, luma, luma_pitch, chroma,
                                chroma_pitch, width, height, device_ordinal,
                                distance, qp, out_file, stats);
}

}  // namespace cujpegxl
