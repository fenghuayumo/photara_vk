#include "photara_vk/command.hpp"

#include "device_impl.hpp"
#include "photara_vk/check.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace photara::vk {
namespace {

constexpr std::uint32_t kDescriptorSets = 512;
constexpr std::uint32_t kDescriptorsPerSet = 16;

}  // namespace

CommandEncoder::CommandEncoder(
    std::shared_ptr<Device::Impl> impl, const BarrierPolicy policy)
    : impl_(std::move(impl)), policy_(policy) {
    if (!impl_ || impl_->device == VK_NULL_HANDLE) {
        throw std::logic_error("photara_vk Device is empty");
    }
    {
        std::lock_guard lock(impl_->pool_mutex);
        VkCommandBufferAllocateInfo allocate{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = impl_->command_pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        check_vk(vkAllocateCommandBuffers(impl_->device, &allocate, &command_),
                 "vkAllocateCommandBuffers");
    }
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check_vk(vkCreateFence(impl_->device, &fence_info, nullptr, &fence_),
             "vkCreateFence");
    if (!impl_->caps.enabled.push_descriptors) {
        add_descriptor_pool(kDescriptorSets);
    }
}

CommandEncoder::~CommandEncoder() {
    if (!impl_ || impl_->device == VK_NULL_HANDLE) return;
    try {
        if (recording_ || in_flight_) submit_wait();
    } catch (...) {
    }
    for (const VkDescriptorPool pool : pools_) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(impl_->device, pool, nullptr);
        }
    }
    pools_.clear();
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(impl_->device, fence_, nullptr);
    }
    if (command_ != VK_NULL_HANDLE) {
        std::lock_guard lock(impl_->pool_mutex);
        vkFreeCommandBuffers(impl_->device, impl_->command_pool, 1, &command_);
    }
}

CommandEncoder::CommandEncoder(CommandEncoder&& other) noexcept {
    *this = std::move(other);
}

CommandEncoder& CommandEncoder::operator=(CommandEncoder&& other) noexcept {
    if (this == &other) return *this;
    std::swap(impl_, other.impl_);
    std::swap(policy_, other.policy_);
    std::swap(command_, other.command_);
    std::swap(fence_, other.fence_);
    std::swap(pools_, other.pools_);
    std::swap(pool_sets_, other.pool_sets_);
    std::swap(pending_sets_, other.pending_sets_);
    std::swap(descriptor_infos_, other.descriptor_infos_);
    std::swap(descriptor_writes_, other.descriptor_writes_);
    std::swap(recording_, other.recording_);
    std::swap(in_flight_, other.in_flight_);
    return *this;
}

void CommandEncoder::retire() {
    if (!in_flight_) return;
    check_vk(vkWaitForFences(impl_->device, 1, &fence_, VK_TRUE, UINT64_MAX),
             "vkWaitForFences");
    check_vk(vkResetFences(impl_->device, 1, &fence_), "vkResetFences");
    in_flight_ = false;
    reset_descriptor_pools();
}

void CommandEncoder::begin() {
    if (recording_) return;
    retire();
    check_vk(vkResetCommandBuffer(command_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command_, &begin_info), "vkBeginCommandBuffer");
    recording_ = true;
}

void CommandEncoder::barrier(
    const VkPipelineStageFlags src_stage, const VkPipelineStageFlags dst_stage,
    const VkAccessFlags src_access, const VkAccessFlags dst_access) {
    begin();
    VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory.srcAccessMask = src_access;
    memory.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(
        command_, src_stage, dst_stage, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

void CommandEncoder::barrier() {
    barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}

void CommandEncoder::after_dispatch() {
    switch (policy_) {
    case BarrierPolicy::none:
        break;
    case BarrierPolicy::after_compute:
        barrier(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_TRANSFER_READ_BIT);
        break;
    case BarrierPolicy::after_compute_indirect:
        barrier(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
        break;
    }
}

void CommandEncoder::push(
    const ComputePipeline& pipeline, const void* push_constants,
    const std::uint32_t push_bytes) {
    if (push_bytes != pipeline.push_bytes()) {
        throw std::invalid_argument("Push constant size does not match pipeline");
    }
    if (push_bytes == 0) return;
    if (push_constants == nullptr) {
        throw std::invalid_argument("Push constants pointer is null");
    }
    vkCmdPushConstants(
        command_, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes,
        push_constants);
}

void CommandEncoder::add_descriptor_pool(const std::uint32_t sets) {
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets * kDescriptorsPerSet},
    };
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = sets;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    check_vk(vkCreateDescriptorPool(impl_->device, &pool_info, nullptr, &pool),
             "vkCreateDescriptorPool");
    pools_.push_back(pool);
    pool_sets_ = sets;
}

void CommandEncoder::reset_descriptor_pools() {
    pending_sets_.clear();
    if (pools_.empty()) return;
    for (std::size_t index = 0; index + 1 < pools_.size(); ++index) {
        vkDestroyDescriptorPool(impl_->device, pools_[index], nullptr);
    }
    const VkDescriptorPool keep = pools_.back();
    pools_.clear();
    pools_.push_back(keep);
    check_vk(vkResetDescriptorPool(impl_->device, keep, 0),
             "vkResetDescriptorPool");
}

VkDescriptorSet CommandEncoder::allocate_set(const VkDescriptorSetLayout layout) {
    if (pools_.empty()) add_descriptor_pool(kDescriptorSets);
    for (int attempt = 0; attempt < 2; ++attempt) {
        VkDescriptorSetAllocateInfo allocate{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pools_.back();
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult result =
            vkAllocateDescriptorSets(impl_->device, &allocate, &set);
        if (result == VK_SUCCESS) {
            pending_sets_.push_back(set);
            return set;
        }
        const bool exhausted = result == VK_ERROR_OUT_OF_POOL_MEMORY ||
                               result == VK_ERROR_FRAGMENTED_POOL;
        if (!exhausted || attempt == 1 || pool_sets_ >= (1u << 20)) {
            check_vk(result, "vkAllocateDescriptorSets");
        }
        add_descriptor_pool(std::max(kDescriptorSets, pool_sets_ * 2));
    }
    throw std::runtime_error("vkAllocateDescriptorSets failed");
}

void CommandEncoder::bind_storage(
    const ComputePipeline& pipeline, std::span<const BufferBinding> bindings) {
    if (bindings.size() != pipeline.binding_count()) {
        throw std::invalid_argument("Descriptor binding count does not match pipeline");
    }
    descriptor_infos_.resize(bindings.size());
    descriptor_writes_.resize(bindings.size());
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        descriptor_infos_[index].buffer = bindings[index].buffer;
        descriptor_infos_[index].offset = bindings[index].offset;
        descriptor_infos_[index].range = bindings[index].range;
        descriptor_writes_[index] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        descriptor_writes_[index].dstBinding = index;
        descriptor_writes_[index].descriptorCount = 1;
        descriptor_writes_[index].descriptorType =
            pipeline.descriptor_type(index);
        descriptor_writes_[index].pBufferInfo = &descriptor_infos_[index];
    }
    vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    if (pipeline.push_descriptors() && impl_->cmd_push_descriptor != nullptr) {
        impl_->cmd_push_descriptor(
            command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0,
            static_cast<std::uint32_t>(descriptor_writes_.size()),
            descriptor_writes_.data());
        return;
    }
    const VkDescriptorSet set = allocate_set(pipeline.set_layout());
    for (VkWriteDescriptorSet& write : descriptor_writes_) write.dstSet = set;
    vkUpdateDescriptorSets(
        impl_->device, static_cast<std::uint32_t>(descriptor_writes_.size()),
        descriptor_writes_.data(), 0, nullptr);
    vkCmdBindDescriptorSets(
        command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, &set, 0,
        nullptr);
}

void CommandEncoder::dispatch(
    const ComputePipeline& pipeline, std::span<const BufferBinding> bindings,
    const void* push_constants, const std::uint32_t push_bytes,
    const std::uint32_t groups_x, const std::uint32_t groups_y,
    const std::uint32_t groups_z) {
    if (groups_x == 0 || groups_y == 0 || groups_z == 0) return;
    begin();
    bind_storage(pipeline, bindings);
    push(pipeline, push_constants, push_bytes);
    vkCmdDispatch(command_, groups_x, groups_y, groups_z);
    after_dispatch();
}

void CommandEncoder::dispatch_indirect(
    const ComputePipeline& pipeline, std::span<const BufferBinding> bindings,
    const void* push_constants, const std::uint32_t push_bytes,
    const BufferBinding indirect) {
    if (indirect.buffer == VK_NULL_HANDLE) {
        throw std::invalid_argument("Indirect dispatch buffer is null");
    }
    begin();
    bind_storage(pipeline, bindings);
    push(pipeline, push_constants, push_bytes);
    barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    vkCmdDispatchIndirect(command_, indirect.buffer, indirect.offset);
    after_dispatch();
}

void CommandEncoder::dispatch_indirect(
    const ComputePipeline& pipeline, std::span<const BufferBinding> bindings,
    const void* push_constants, const std::uint32_t push_bytes,
    const Buffer& indirect, const VkDeviceSize offset) {
    dispatch_indirect(
        pipeline, bindings, push_constants, push_bytes,
        binding(indirect, offset, sizeof(VkDispatchIndirectCommand)));
}

void CommandEncoder::fill_u32(
    const Buffer& destination, const VkDeviceSize offset, const VkDeviceSize bytes,
    const std::uint32_t value) {
    if (bytes == 0) return;
    begin();
    vkCmdFillBuffer(
        command_, destination.handle(), destination.offset() + offset, bytes, value);
    barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
}

void CommandEncoder::copy(
    const Buffer& destination, const VkDeviceSize destination_offset,
    const Buffer& source, const VkDeviceSize source_offset,
    const VkDeviceSize bytes) {
    if (bytes == 0) return;
    begin();
    VkBufferCopy region{};
    region.srcOffset = source.offset() + source_offset;
    region.dstOffset = destination.offset() + destination_offset;
    region.size = bytes;
    vkCmdCopyBuffer(command_, source.handle(), destination.handle(), 1, &region);
    barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
}

void CommandEncoder::submit() {
    if (!recording_) return;
    check_vk(vkEndCommandBuffer(command_), "vkEndCommandBuffer");
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_;
    {
        std::lock_guard lock(impl_->queue_mutex);
        check_vk(vkQueueSubmit(impl_->queue, 1, &submit_info, fence_), "vkQueueSubmit");
    }
    recording_ = false;
    in_flight_ = true;
}

void CommandEncoder::submit_wait() {
    submit();
    retire();
}

void CommandEncoder::wait() { retire(); }

VkCommandBuffer CommandEncoder::native() {
    begin();
    return command_;
}

CommandEncoder Device::encoder(const BarrierPolicy policy) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return CommandEncoder(impl_, policy);
}

EncoderRing::EncoderRing(
    Device device, const std::uint32_t slots, const BarrierPolicy policy)
    : device_(std::move(device)) {
    if (!device_.valid()) throw std::logic_error("photara_vk Device is empty");
    if (slots == 0) throw std::invalid_argument("EncoderRing needs at least one slot");
    slots_.reserve(slots);
    for (std::uint32_t i = 0; i < slots; ++i) {
        slots_.push_back(device_.encoder(policy));
    }
}

EncoderRing::EncoderRing(EncoderRing&& other) noexcept {
    *this = std::move(other);
}

EncoderRing& EncoderRing::operator=(EncoderRing&& other) noexcept {
    if (this == &other) return *this;
    std::swap(device_, other.device_);
    std::swap(slots_, other.slots_);
    std::swap(index_, other.index_);
    return *this;
}

CommandEncoder& EncoderRing::acquire() {
    if (slots_.empty()) throw std::logic_error("EncoderRing has no slots");
    slots_[index_].begin();
    return slots_[index_];
}

void EncoderRing::submit() {
    if (slots_.empty() || !slots_[index_].recording()) return;
    slots_[index_].submit();
    index_ = (index_ + 1U) % static_cast<std::uint32_t>(slots_.size());
}

void EncoderRing::submit_wait() {
    if (slots_.empty() || !slots_[index_].recording()) return;
    slots_[index_].submit_wait();
    index_ = (index_ + 1U) % static_cast<std::uint32_t>(slots_.size());
}

bool EncoderRing::recording() const {
    return !slots_.empty() && slots_[index_].recording();
}

CommandEncoder& EncoderRing::at(const std::uint32_t slot) {
    if (slot >= slots_.size()) throw std::out_of_range("EncoderRing slot");
    return slots_[slot];
}

const CommandEncoder& EncoderRing::at(const std::uint32_t slot) const {
    if (slot >= slots_.size()) throw std::out_of_range("EncoderRing slot");
    return slots_[slot];
}

}  // namespace photara::vk
