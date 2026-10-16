#include "photara_vk/query.hpp"

#include "device_impl.hpp"
#include "photara_vk/device.hpp"
#include "photara_vk/check.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace photara::vk {
TimestampPool::TimestampPool(
    VkDevice device, const std::uint32_t timestamp_count,
    const double timestamp_period_nanoseconds,
    const std::uint32_t timestamp_valid_bits)
    : device_(device), count_(timestamp_count),
      timestamp_period_nanoseconds_(timestamp_period_nanoseconds),
      timestamp_valid_bits_(timestamp_valid_bits) {
    if (device_ == VK_NULL_HANDLE) {
        throw std::invalid_argument("TimestampPool requires a Vulkan device");
    }
    if (count_ == 0) throw std::invalid_argument("TimestampPool needs one query");
    if (!(timestamp_period_nanoseconds_ > 0.0)) {
        throw std::invalid_argument("Vulkan timestamp period must be positive");
    }
    if (timestamp_valid_bits_ == 0) {
        throw std::invalid_argument("Vulkan queue family has no timestamp support");
    }

    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = count_;
    check_vk(vkCreateQueryPool(device_, &info, nullptr, &pool_), "vkCreateQueryPool");
}

TimestampPool::~TimestampPool() {
    if (device_ != VK_NULL_HANDLE && pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_, pool_, nullptr);
    }
}

TimestampPool::TimestampPool(TimestampPool&& other) noexcept {
    *this = std::move(other);
}

TimestampPool& TimestampPool::operator=(TimestampPool&& other) noexcept {
    if (this == &other) return *this;
    if (device_ != VK_NULL_HANDLE && pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_, pool_, nullptr);
    }
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    pool_ = std::exchange(other.pool_, VK_NULL_HANDLE);
    count_ = std::exchange(other.count_, 0U);
    timestamp_period_nanoseconds_ =
        std::exchange(other.timestamp_period_nanoseconds_, 1.0);
    timestamp_valid_bits_ = std::exchange(other.timestamp_valid_bits_, 64U);
    return *this;
}

std::vector<std::uint64_t> TimestampPool::results(const bool wait) const {
    if (pool_ == VK_NULL_HANDLE) {
        throw std::logic_error("TimestampPool is empty");
    }
    std::vector<std::uint64_t> values(count_);
    const VkQueryResultFlags flags =
        VK_QUERY_RESULT_64_BIT | (wait ? VK_QUERY_RESULT_WAIT_BIT : 0);
    const VkResult result = vkGetQueryPoolResults(
        device_, pool_, 0, count_, values.size() * sizeof(values[0]), values.data(),
        sizeof(values[0]), flags);
    if (result == VK_NOT_READY && !wait) return {};
    check_vk(result, "vkGetQueryPoolResults");
    return values;
}

std::uint64_t TimestampPool::result(
    const std::uint32_t index, const bool wait) const {
    if (index >= count_) throw std::out_of_range("TimestampPool query");
    std::uint64_t value = 0;
    const VkQueryResultFlags flags =
        VK_QUERY_RESULT_64_BIT | (wait ? VK_QUERY_RESULT_WAIT_BIT : 0);
    const VkResult result = vkGetQueryPoolResults(
        device_, pool_, index, 1, sizeof(value), &value, sizeof(value), flags);
    if (result == VK_NOT_READY && !wait) return 0;
    check_vk(result, "vkGetQueryPoolResults");
    return value;
}

double TimestampPool::elapsed_nanoseconds(
    const std::uint32_t start, const std::uint32_t end) const {
    const std::uint64_t first = result(start);
    const std::uint64_t second = result(end);
    std::uint64_t mask = std::numeric_limits<std::uint64_t>::max();
    if (timestamp_valid_bits_ < 64) {
        mask = (std::uint64_t{1} << timestamp_valid_bits_) - 1;
    }
    return double((second - first) & mask) * timestamp_period_nanoseconds_;
}

double TimestampPool::elapsed_seconds(
    const std::uint32_t start, const std::uint32_t end) const {
    return elapsed_nanoseconds(start, end) * 1.0e-9;
}

TimestampPool Device::create_timestamp_pool(
    const std::uint32_t timestamp_count) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return TimestampPool(
        impl_->device, timestamp_count, impl_->caps.timestamp_period_nanoseconds,
        impl_->caps.timestamp_valid_bits);
}

}  // namespace photara::vk
