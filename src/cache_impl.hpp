#pragma once
#include <vulkan/vulkan.h>

#include <mutex>

#include "photara_vk/cache.hpp"

namespace photara::vk {
struct PipelineCache::Impl {
    std::shared_ptr<void> keep;
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineCache handle = VK_NULL_HANDLE;
    mutable std::mutex mutex;
    ~Impl() {
        if (handle != VK_NULL_HANDLE) vkDestroyPipelineCache(device, handle, nullptr);
    }
};
}  // namespace photara::vk
