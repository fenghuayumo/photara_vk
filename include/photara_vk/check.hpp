#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <vulkan/vulkan.h>

namespace photara::vk {

inline void check_vk(const VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return;
    throw std::runtime_error(
        std::string(operation) + " failed with VkResult " +
        std::to_string(static_cast<int>(result)));
}

inline std::uint32_t ceil_div(const std::uint32_t value, const std::uint32_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1U) / divisor;
}

}  // namespace photara::vk
