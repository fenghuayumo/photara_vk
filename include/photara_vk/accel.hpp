#pragma once

#include "photara_vk/buffer.hpp"

#include <cstdint>
#include <memory>

#include <vulkan/vulkan.h>

namespace photara::vk {

class Device;

// One bottom-level triangle mesh and one instance in a top-level structure.
// The vertex, index, and instance buffers stay alive for the scene: the
// acceleration structures address them. This is the compute ray-query path.
// Ray-tracing pipelines and shader binding tables are not created here.
class TriangleScene {
public:
    TriangleScene() = default;
    ~TriangleScene();
    TriangleScene(TriangleScene&& other) noexcept;
    TriangleScene& operator=(TriangleScene&& other) noexcept;
    TriangleScene(const TriangleScene&) = delete;
    TriangleScene& operator=(const TriangleScene&) = delete;

    [[nodiscard]] bool valid() const noexcept { return top_level_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkAccelerationStructureKHR top_level() const noexcept {
        return top_level_;
    }
    [[nodiscard]] VkAccelerationStructureKHR bottom_level() const noexcept {
        return bottom_level_;
    }

private:
    friend class Device;

    TriangleScene(
        std::shared_ptr<void> device_keep, VkDevice device, Buffer vertices,
        Buffer indices, Buffer instances, Buffer bottom_storage, Buffer top_storage,
        VkAccelerationStructureKHR bottom_level, VkAccelerationStructureKHR top_level,
        PFN_vkDestroyAccelerationStructureKHR destroy);

    std::shared_ptr<void> device_keep_;
    VkDevice device_ = VK_NULL_HANDLE;
    Buffer vertices_;
    Buffer indices_;
    Buffer instances_;
    Buffer bottom_storage_;
    Buffer top_storage_;
    VkAccelerationStructureKHR bottom_level_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR top_level_ = VK_NULL_HANDLE;
    PFN_vkDestroyAccelerationStructureKHR destroy_ = nullptr;
};

}  // namespace photara::vk
