#pragma once

#include "photara_vk/device.hpp"

#include <mutex>

#include <vulkan/vulkan.h>

namespace photara::vk {

struct Device::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    Capabilities caps{};
    bool owns_instance = false;
    bool owns_device = false;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    PFN_vkCmdPushDescriptorSetKHR cmd_push_descriptor = nullptr;
    mutable std::mutex pool_mutex;
    VkPhysicalDeviceMemoryProperties memory{};
    VkDeviceSize non_coherent_atom_size = 1;

    void ensure_command_pool();
    ~Impl();
};

}  // namespace photara::vk
