#pragma once
#include "photara_vk/buffer.hpp"
#include <cstdint>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace photara::vk {
class Device;
struct TriangleSceneOptions {
    VkGeometryFlagsKHR geometry_flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    VkBuildAccelerationStructureFlagsKHR build_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
};

// An object-space BLAS. Shared between any number of TLAS instances/scenes.
class BottomLevel {
public:
    ~BottomLevel();
    BottomLevel(const BottomLevel&) = delete;
    BottomLevel& operator=(const BottomLevel&) = delete;
    [[nodiscard]] VkAccelerationStructureKHR handle() const noexcept { return handle_; }
    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] VkDeviceAddress address() const noexcept { return address_; }
private:
    friend class Device;
    BottomLevel(std::shared_ptr<void> keep, VkDevice device, Buffer vertices, Buffer indices,
        Buffer storage, VkAccelerationStructureKHR handle, VkDeviceAddress address,
        PFN_vkDestroyAccelerationStructureKHR destroy);
    std::shared_ptr<void> keep_;
    VkDevice device_ = VK_NULL_HANDLE;
    Buffer vertices_, indices_, storage_;
    VkAccelerationStructureKHR handle_ = VK_NULL_HANDLE;
    VkDeviceAddress address_ = 0;
    PFN_vkDestroyAccelerationStructureKHR destroy_ = nullptr;
};
struct TriangleInstance {
    std::shared_ptr<const BottomLevel> geometry;
    VkTransformMatrixKHR transform = {{{1,0,0,0},{0,1,0,0},{0,0,1,0}}};
    std::uint32_t custom_index = 0;
    std::uint8_t mask = 0xff;
    VkGeometryInstanceFlagsKHR flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
};
// TLAS plus strong references to its BLASes. Device and geometry remain alive
// even when the caller releases its original handles before this scene.
class TriangleScene {
public:
    TriangleScene() = default;
    ~TriangleScene();
    TriangleScene(TriangleScene&& other) noexcept;
    TriangleScene& operator=(TriangleScene&& other) noexcept;
    TriangleScene(const TriangleScene&) = delete;
    TriangleScene& operator=(const TriangleScene&) = delete;
    [[nodiscard]] bool valid() const noexcept { return top_level_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkAccelerationStructureKHR top_level() const noexcept { return top_level_; }
    [[nodiscard]] VkAccelerationStructureKHR bottom_level() const noexcept {
        return geometry_.empty() ? VK_NULL_HANDLE : geometry_[0]->handle();
    }
private:
    friend class Device;
    TriangleScene(std::shared_ptr<void> keep, VkDevice device,
        std::vector<std::shared_ptr<const BottomLevel>> geometry, Buffer instances,
        Buffer storage, VkAccelerationStructureKHR top, PFN_vkDestroyAccelerationStructureKHR destroy);
    std::shared_ptr<void> keep_;
    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<const BottomLevel>> geometry_;
    Buffer instances_, storage_, update_scratch_;
    VkBuildAccelerationStructureFlagsKHR build_flags_ = 0;
    VkAccelerationStructureKHR top_level_ = VK_NULL_HANDLE;
    PFN_vkDestroyAccelerationStructureKHR destroy_ = nullptr;
};
} // namespace photara::vk
