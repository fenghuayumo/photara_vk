#include "photara_vk/accel.hpp"

#include "device_impl.hpp"
#include "photara_vk/check.hpp"

#include <stdexcept>
#include <utility>

namespace photara::vk {
namespace {

struct AccelFns {
    PFN_vkCreateAccelerationStructureKHR create = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroy = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR sizes = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR build = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR address = nullptr;
};

AccelFns load_accel(const VkDevice device) {
    AccelFns fns;
    fns.create = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    fns.destroy = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
    fns.sizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    fns.build = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
    fns.address = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
    if (fns.create == nullptr || fns.destroy == nullptr || fns.sizes == nullptr ||
        fns.build == nullptr || fns.address == nullptr) {
        throw std::runtime_error(
            "Required Vulkan acceleration-structure entry points are unavailable");
    }
    return fns;
}

Buffer make_buffer(
    const Device::Impl& impl, const VkDeviceSize bytes, const MemoryKind kind,
    const VkBufferUsageFlags usage) {
    return Buffer(impl.physical, impl.device, bytes, kind, usage, true);
}

struct BuiltLevel {
    Buffer storage;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
};

BuiltLevel build_level(
    Device::Impl& impl, const AccelFns& fns, const VkAccelerationStructureTypeKHR type,
    const VkAccelerationStructureGeometryKHR& geometry, const std::uint32_t primitives) {
    VkAccelerationStructureBuildGeometryInfoKHR build_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build_info.type = type;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries = &geometry;
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    fns.sizes(
        impl.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info,
        &primitives, &sizes);
    if (sizes.accelerationStructureSize == 0) {
        throw std::runtime_error("Vulkan acceleration structure size is zero");
    }

    BuiltLevel level;
    level.storage = make_buffer(
        impl, sizes.accelerationStructureSize, MemoryKind::device_local,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    VkAccelerationStructureCreateInfoKHR create_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    create_info.buffer = level.storage.handle();
    create_info.size = sizes.accelerationStructureSize;
    create_info.type = type;
    check_vk(
        fns.create(impl.device, &create_info, nullptr, &level.handle),
        "vkCreateAccelerationStructureKHR");

    auto scratch = make_buffer(
        impl, sizes.buildScratchSize, MemoryKind::device_local,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    build_info.dstAccelerationStructure = level.handle;
    build_info.scratchData.deviceAddress = scratch.device_address();
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = primitives;
    const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&range};

    impl.ensure_command_pool();
    VkCommandBuffer command = VK_NULL_HANDLE;
    {
        std::lock_guard lock(impl.pool_mutex);
        VkCommandBufferAllocateInfo allocate{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = impl.command_pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        check_vk(
            vkAllocateCommandBuffers(impl.device, &allocate, &command),
            "vkAllocateCommandBuffers");
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
    fns.build(command, 1, &build_info, ranges);
    check_vk(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    {
        std::lock_guard lock(queue_mutex(impl.queue));
        check_vk(vkQueueSubmit(impl.queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
        check_vk(vkQueueWaitIdle(impl.queue), "vkQueueWaitIdle");
    }
    {
        std::lock_guard lock(impl.pool_mutex);
        vkFreeCommandBuffers(impl.device, impl.command_pool, 1, &command);
    }
    return level;
}

}  // namespace

TriangleScene::TriangleScene(
    std::shared_ptr<void> device_keep, const VkDevice device, Buffer vertices,
    Buffer indices, Buffer instances, Buffer bottom_storage, Buffer top_storage,
    const VkAccelerationStructureKHR bottom_level,
    const VkAccelerationStructureKHR top_level,
    const PFN_vkDestroyAccelerationStructureKHR destroy)
    : device_keep_(std::move(device_keep)),
      device_(device),
      vertices_(std::move(vertices)),
      indices_(std::move(indices)),
      instances_(std::move(instances)),
      bottom_storage_(std::move(bottom_storage)),
      top_storage_(std::move(top_storage)),
      bottom_level_(bottom_level),
      top_level_(top_level),
      destroy_(destroy) {}

TriangleScene::~TriangleScene() {
    if (device_ != VK_NULL_HANDLE && destroy_ != nullptr) {
        if (top_level_ != VK_NULL_HANDLE) destroy_(device_, top_level_, nullptr);
        if (bottom_level_ != VK_NULL_HANDLE) destroy_(device_, bottom_level_, nullptr);
    }
    top_level_ = VK_NULL_HANDLE;
    bottom_level_ = VK_NULL_HANDLE;
}

TriangleScene::TriangleScene(TriangleScene&& other) noexcept {
    *this = std::move(other);
}

TriangleScene& TriangleScene::operator=(TriangleScene&& other) noexcept {
    if (this == &other) return *this;
    std::swap(device_keep_, other.device_keep_);
    std::swap(device_, other.device_);
    std::swap(vertices_, other.vertices_);
    std::swap(indices_, other.indices_);
    std::swap(instances_, other.instances_);
    std::swap(bottom_storage_, other.bottom_storage_);
    std::swap(top_storage_, other.top_storage_);
    std::swap(bottom_level_, other.bottom_level_);
    std::swap(top_level_, other.top_level_);
    std::swap(destroy_, other.destroy_);
    return *this;
}

TriangleScene Device::create_triangle_scene(
    const std::span<const float> positions,
    const std::span<const std::uint32_t> indices) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    if (!impl_->caps.enabled.ray_query || !impl_->caps.enabled.buffer_device_address) {
        throw std::runtime_error(
            "The selected Vulkan device does not support ray queries");
    }
    if (positions.empty() || positions.size() % 3 != 0 || indices.empty() ||
        indices.size() % 3 != 0) {
        throw std::invalid_argument(
            "Ray-query geometry must contain float3 positions and triangle indices");
    }

    const AccelFns fns = load_accel(impl_->device);
    const auto input_usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    auto vertices = make_buffer(
        *impl_, positions.size_bytes(), MemoryKind::host_visible, input_usage);
    auto index_buffer = make_buffer(
        *impl_, indices.size_bytes(), MemoryKind::host_visible, input_usage);
    vertices.upload(positions.data(), positions.size_bytes());
    index_buffer.upload(indices.data(), indices.size_bytes());

    VkAccelerationStructureGeometryTrianglesDataKHR triangle_data{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangle_data.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangle_data.vertexData.deviceAddress = vertices.device_address();
    triangle_data.vertexStride = sizeof(float) * 3;
    triangle_data.maxVertex = static_cast<std::uint32_t>(positions.size() / 3 - 1);
    triangle_data.indexType = VK_INDEX_TYPE_UINT32;
    triangle_data.indexData.deviceAddress = index_buffer.device_address();
    VkAccelerationStructureGeometryKHR bottom_geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    bottom_geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    bottom_geometry.geometry.triangles = triangle_data;
    bottom_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    BuiltLevel bottom = build_level(
        *impl_, fns, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, bottom_geometry,
        static_cast<std::uint32_t>(indices.size() / 3));

    VkAccelerationStructureDeviceAddressInfoKHR address_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address_info.accelerationStructure = bottom.handle;
    VkAccelerationStructureInstanceKHR geometry_instance{};
    geometry_instance.transform.matrix[0][0] = 1.0F;
    geometry_instance.transform.matrix[1][1] = 1.0F;
    geometry_instance.transform.matrix[2][2] = 1.0F;
    geometry_instance.mask = 0xff;
    geometry_instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    geometry_instance.accelerationStructureReference =
        fns.address(impl_->device, &address_info);
    auto instances = make_buffer(
        *impl_, sizeof(geometry_instance), MemoryKind::host_visible, input_usage);
    instances.upload(&geometry_instance, sizeof(geometry_instance));

    VkAccelerationStructureGeometryInstancesDataKHR instance_data{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instance_data.arrayOfPointers = VK_FALSE;
    instance_data.data.deviceAddress = instances.device_address();
    VkAccelerationStructureGeometryKHR top_geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    top_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    top_geometry.geometry.instances = instance_data;
    top_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    BuiltLevel top = build_level(
        *impl_, fns, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, top_geometry, 1);

    std::shared_ptr<void> keep(impl_, impl_.get());
    return TriangleScene(
        std::move(keep), impl_->device, std::move(vertices), std::move(index_buffer),
        std::move(instances), std::move(bottom.storage), std::move(top.storage),
        bottom.handle, top.handle, fns.destroy);
}

}  // namespace photara::vk
