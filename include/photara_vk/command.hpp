#pragma once

#include "photara_vk/buffer.hpp"
#include "photara_vk/device.hpp"
#include "photara_vk/pipeline.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

namespace photara::vk {

struct BufferBinding {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = VK_WHOLE_SIZE;
};

[[nodiscard]] inline BufferBinding binding(
    const Buffer& buffer, const VkDeviceSize offset = 0,
    const VkDeviceSize range = VK_WHOLE_SIZE) {
    BufferBinding result;
    result.buffer = buffer.handle();
    result.offset = buffer.offset() + offset;
    result.range = range == VK_WHOLE_SIZE ? buffer.size() - offset : range;
    return result;
}

[[nodiscard]] inline BufferBinding binding(const VkDescriptorBufferInfo& info) {
    BufferBinding result;
    result.buffer = info.buffer;
    result.offset = info.offset;
    result.range = info.range;
    return result;
}

class CommandEncoder {
public:
    CommandEncoder() = default;
    CommandEncoder(std::shared_ptr<Device::Impl> impl, BarrierPolicy policy);
    ~CommandEncoder();
    CommandEncoder(CommandEncoder&& other) noexcept;
    CommandEncoder& operator=(CommandEncoder&& other) noexcept;
    CommandEncoder(const CommandEncoder&) = delete;
    CommandEncoder& operator=(const CommandEncoder&) = delete;

    void dispatch(
        const ComputePipeline& pipeline, std::span<const BufferBinding> bindings,
        const void* push_constants, std::uint32_t push_bytes,
        std::uint32_t groups_x, std::uint32_t groups_y = 1,
        std::uint32_t groups_z = 1);
    void dispatch_indirect(
        const ComputePipeline& pipeline, std::span<const BufferBinding> bindings,
        const void* push_constants, std::uint32_t push_bytes,
        BufferBinding indirect);
    void dispatch_indirect(
        const ComputePipeline& pipeline, std::span<const BufferBinding> bindings,
        const void* push_constants, std::uint32_t push_bytes,
        const Buffer& indirect, VkDeviceSize offset);
    void fill_u32(
        const Buffer& destination, VkDeviceSize offset, VkDeviceSize bytes,
        std::uint32_t value);
    void copy(
        const Buffer& destination, VkDeviceSize destination_offset,
        const Buffer& source, VkDeviceSize source_offset, VkDeviceSize bytes);
    void barrier();
    void barrier(
        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
        VkAccessFlags src_access, VkAccessFlags dst_access);
    void submit();
    void submit_wait();
    void wait();

    [[nodiscard]] VkCommandBuffer native();
    [[nodiscard]] VkFence fence() const noexcept { return fence_; }
    [[nodiscard]] bool recording() const noexcept { return recording_; }
    [[nodiscard]] bool in_flight() const noexcept { return in_flight_; }

private:
    void begin();
    void retire();
    void bind_storage(
        const ComputePipeline& pipeline, std::span<const BufferBinding> bindings);
    void push(
        const ComputePipeline& pipeline, const void* push_constants,
        std::uint32_t push_bytes);
    void after_dispatch();
    void add_descriptor_pool(std::uint32_t sets);
    void reset_descriptor_pools();
    [[nodiscard]] VkDescriptorSet allocate_set(VkDescriptorSetLayout layout);

    friend class EncoderRing;

    std::shared_ptr<Device::Impl> impl_;
    BarrierPolicy policy_ = BarrierPolicy::after_compute;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    // Descriptor sets stay alive until the submission that uses them retires.
    // A batch that outgrows the current pool allocates another one instead of
    // resetting a pool a recorded command still references.
    std::vector<VkDescriptorPool> pools_;
    std::uint32_t pool_sets_ = 0;
    std::vector<VkDescriptorSet> pending_sets_;
    // Reused across dispatches so descriptor binding does not allocate two
    // temporary vectors for every recorded operation.
    std::vector<VkDescriptorBufferInfo> descriptor_infos_;
    std::vector<VkWriteDescriptorSet> descriptor_writes_;
    bool recording_ = false;
    bool in_flight_ = false;
};

class EncoderRing {
public:
    EncoderRing() = default;
    EncoderRing(
        Device device, std::uint32_t slots,
        BarrierPolicy policy = BarrierPolicy::after_compute_indirect);
    EncoderRing(EncoderRing&& other) noexcept;
    EncoderRing& operator=(EncoderRing&& other) noexcept;
    EncoderRing(const EncoderRing&) = delete;
    EncoderRing& operator=(const EncoderRing&) = delete;
    ~EncoderRing() = default;

    CommandEncoder& acquire();
    void submit();
    void submit_wait();

    [[nodiscard]] bool recording() const;
    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }
    [[nodiscard]] std::uint32_t size() const noexcept {
        return static_cast<std::uint32_t>(slots_.size());
    }
    [[nodiscard]] CommandEncoder& at(std::uint32_t slot);
    [[nodiscard]] const CommandEncoder& at(std::uint32_t slot) const;

private:
    Device device_;
    std::vector<CommandEncoder> slots_;
    std::uint32_t index_ = 0;
};

}  // namespace photara::vk
