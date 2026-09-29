#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

namespace photara::vk {

class ComputePipeline {
public:
    ComputePipeline() = default;
    ComputePipeline(
        VkDevice logical, std::span<const std::byte> spirv,
        std::span<const VkDescriptorType> descriptor_types,
        std::uint32_t push_bytes, bool push_descriptors);
    ComputePipeline(
        VkDevice logical, std::span<const std::byte> spirv,
        std::uint32_t storage_buffer_count, std::uint32_t push_bytes,
        bool push_descriptors);
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    [[nodiscard]] bool valid() const noexcept { return handle_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkPipeline handle() const noexcept { return handle_; }
    [[nodiscard]] VkPipelineLayout layout() const noexcept { return layout_; }
    [[nodiscard]] VkDescriptorSetLayout set_layout() const noexcept {
        return set_layout_;
    }
    [[nodiscard]] std::uint32_t binding_count() const noexcept {
        return static_cast<std::uint32_t>(types_.size());
    }
    [[nodiscard]] std::uint32_t push_bytes() const noexcept { return push_bytes_; }
    [[nodiscard]] bool push_descriptors() const noexcept {
        return push_descriptors_;
    }
    [[nodiscard]] VkDescriptorType descriptor_type(std::uint32_t index) const {
        return types_[index];
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline handle_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorType> types_;
    std::uint32_t push_bytes_ = 0;
    bool push_descriptors_ = false;
};

}  // namespace photara::vk
