#include "photara_vk/pipeline.hpp"

#include "device_impl.hpp"
#include "photara_vk/check.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace photara::vk {

ComputePipeline::ComputePipeline(
    VkDevice logical, std::span<const std::byte> spirv,
    std::span<const VkDescriptorType> descriptor_types,
    const std::uint32_t push_bytes, const bool push_descriptors)
    : device_(logical),
      types_(descriptor_types.begin(), descriptor_types.end()),
      push_bytes_(push_bytes),
      push_descriptors_(push_descriptors) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(types_.size());
    for (std::uint32_t index = 0; index < types_.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = types_[index];
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    if (push_descriptors_) {
        layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }
    layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout_info.pBindings = bindings.empty() ? nullptr : bindings.data();
    check_vk(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &set_layout_),
             "vkCreateDescriptorSetLayout");

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.size = push_bytes_;
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout_;
    pipeline_layout_info.pushConstantRangeCount = push_bytes_ == 0 ? 0U : 1U;
    pipeline_layout_info.pPushConstantRanges =
        push_bytes_ == 0 ? nullptr : &push_range;
    check_vk(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &layout_),
             "vkCreatePipelineLayout");

    if (spirv.empty() || spirv.size() % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error("Invalid embedded SPIR-V bytecode");
    }
    std::vector<std::uint32_t> words(spirv.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), spirv.data(), spirv.size());
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = words.size() * sizeof(std::uint32_t);
    shader_info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    check_vk(vkCreateShaderModule(device_, &shader_info, nullptr, &module),
             "vkCreateShaderModule");

    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage;
    pipeline_info.layout = layout_;
    const VkResult result = vkCreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &handle_);
    vkDestroyShaderModule(device_, module, nullptr);
    check_vk(result, "vkCreateComputePipelines");
}

ComputePipeline::ComputePipeline(
    VkDevice logical, std::span<const std::byte> spirv,
    const std::uint32_t storage_buffer_count, const std::uint32_t push_bytes,
    const bool push_descriptors)
    : ComputePipeline() {
    std::vector<VkDescriptorType> types(
        storage_buffer_count, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    *this = ComputePipeline(logical, spirv, types, push_bytes, push_descriptors);
}

ComputePipeline::~ComputePipeline() {
    if (device_ == VK_NULL_HANDLE) return;
    if (handle_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, handle_, nullptr);
    if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    }
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept {
    *this = std::move(other);
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
    if (this == &other) return *this;
    std::swap(device_, other.device_);
    std::swap(set_layout_, other.set_layout_);
    std::swap(layout_, other.layout_);
    std::swap(handle_, other.handle_);
    std::swap(types_, other.types_);
    std::swap(push_bytes_, other.push_bytes_);
    std::swap(push_descriptors_, other.push_descriptors_);
    return *this;
}

ComputePipeline Device::create_compute(
    std::span<const std::byte> spirv, const std::uint32_t storage_buffer_count,
    const std::uint32_t push_bytes) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return ComputePipeline(
        impl_->device, spirv, storage_buffer_count, push_bytes,
        impl_->caps.enabled.push_descriptors);
}

ComputePipeline Device::create_compute(
    std::span<const std::byte> spirv,
    std::span<const VkDescriptorType> descriptor_types,
    const std::uint32_t push_bytes) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return ComputePipeline(
        impl_->device, spirv, descriptor_types, push_bytes,
        impl_->caps.enabled.push_descriptors);
}

}  // namespace photara::vk
