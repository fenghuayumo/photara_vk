#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace photara::vk {

// A fixed-size VK_QUERY_TYPE_TIMESTAMP pool. Query indices are individual
// timestamps, not start/end pairs; callers choose the intervals they record.
class TimestampPool {
public:
    TimestampPool() = default;
    TimestampPool(
        VkDevice device, std::uint32_t timestamp_count,
        double timestamp_period_nanoseconds, std::uint32_t timestamp_valid_bits);
    ~TimestampPool();
    TimestampPool(TimestampPool&& other) noexcept;
    TimestampPool& operator=(TimestampPool&& other) noexcept;
    TimestampPool(const TimestampPool&) = delete;
    TimestampPool& operator=(const TimestampPool&) = delete;

    [[nodiscard]] bool valid() const noexcept { return pool_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkQueryPool handle() const noexcept { return pool_; }
    [[nodiscard]] std::uint32_t count() const noexcept { return count_; }
    [[nodiscard]] double timestamp_period_nanoseconds() const noexcept {
        return timestamp_period_nanoseconds_;
    }
    [[nodiscard]] std::uint32_t timestamp_valid_bits() const noexcept {
        return timestamp_valid_bits_;
    }

    // Waits for all timestamp results by default. A non-blocking call returns
    // an empty vector when the GPU has not made any result available yet.
    [[nodiscard]] std::vector<std::uint64_t> results(bool wait = true) const;
    [[nodiscard]] std::uint64_t result(std::uint32_t index, bool wait = true) const;
    [[nodiscard]] double elapsed_nanoseconds(
        std::uint32_t start, std::uint32_t end) const;
    [[nodiscard]] double elapsed_seconds(std::uint32_t start, std::uint32_t end) const;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueryPool pool_ = VK_NULL_HANDLE;
    std::uint32_t count_ = 0;
    double timestamp_period_nanoseconds_ = 1.0;
    std::uint32_t timestamp_valid_bits_ = 64;
};

}  // namespace photara::vk
