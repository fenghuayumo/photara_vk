#include "photara_vk/accel.hpp"

#include "device_impl.hpp"
#include "photara_vk/check.hpp"
#include "photara_vk/command.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

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
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkDestroyAccelerationStructureKHR destroy = nullptr;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    BuiltLevel() = default;
    BuiltLevel(const BuiltLevel&) = delete;
    BuiltLevel& operator=(const BuiltLevel&) = delete;
    BuiltLevel(BuiltLevel&& other) noexcept
        : storage(std::move(other.storage)), device(other.device), destroy(other.destroy),
          handle(std::exchange(other.handle, VK_NULL_HANDLE)) {}
    ~BuiltLevel() {
        if (handle != VK_NULL_HANDLE) destroy(device, handle, nullptr);
    }
};

BuiltLevel build_level(
    const std::shared_ptr<Device::Impl>& owner, const AccelFns& fns,
    const VkAccelerationStructureTypeKHR type,
    const VkAccelerationStructureGeometryKHR& geometry, const std::uint32_t primitives,
    const VkBuildAccelerationStructureFlagsKHR flags) {
    auto& impl = *owner;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &properties;
    vkGetPhysicalDeviceProperties2(impl.physical, &properties2);
    if (primitives > properties.maxPrimitiveCount) {
        throw std::invalid_argument("Triangle count exceeds the device acceleration-structure limit");
    }
    VkAccelerationStructureBuildGeometryInfoKHR build_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build_info.type = type;
    build_info.flags = flags;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries = &geometry;
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    fns.sizes(impl.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
              &build_info, &primitives, &sizes);
    if (sizes.accelerationStructureSize == 0) {
        throw std::runtime_error("Vulkan acceleration structure size is zero");
    }
    BuiltLevel level;
    level.device = impl.device;
    level.destroy = fns.destroy;
    level.storage = make_buffer(
        impl, sizes.accelerationStructureSize, MemoryKind::device_local,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    VkAccelerationStructureCreateInfoKHR create_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    create_info.buffer = level.storage.handle();
    create_info.size = sizes.accelerationStructureSize;
    create_info.type = type;
    check_vk(fns.create(impl.device, &create_info, nullptr, &level.handle),
             "vkCreateAccelerationStructureKHR");
    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        1, properties.minAccelerationStructureScratchOffsetAlignment);
    auto scratch = make_buffer(
        impl, sizes.buildScratchSize + alignment - 1, MemoryKind::device_local,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    const VkDeviceAddress base = scratch.device_address();
    build_info.dstAccelerationStructure = level.handle;
    build_info.scratchData.deviceAddress = (base + alignment - 1) & ~(alignment - 1);
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = primitives;
    const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&range};
    auto encoder = CommandEncoder(owner, BarrierPolicy::none);
    const auto command = encoder.native();
    encoder.barrier(VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    fns.build(command, 1, &build_info, ranges);
    encoder.barrier(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    encoder.submit_wait();
    return level;
}

}  // namespace

BottomLevel::BottomLevel(std::shared_ptr<void> keep, VkDevice device, Buffer vertices,
    Buffer indices, Buffer storage, VkAccelerationStructureKHR handle, VkDeviceAddress address,
    PFN_vkDestroyAccelerationStructureKHR destroy)
    : keep_(std::move(keep)), device_(device), vertices_(std::move(vertices)),
      indices_(std::move(indices)), storage_(std::move(storage)), handle_(handle), address_(address), destroy_(destroy) {}
BottomLevel::~BottomLevel() {
    if(handle_ != VK_NULL_HANDLE) destroy_(device_, handle_, nullptr);
}
TriangleScene::TriangleScene(std::shared_ptr<void> keep, VkDevice device,
    std::vector<std::shared_ptr<const BottomLevel>> geometry, Buffer instances,
    Buffer storage, VkAccelerationStructureKHR top, PFN_vkDestroyAccelerationStructureKHR destroy)
    : keep_(std::move(keep)), device_(device), geometry_(std::move(geometry)), instances_(std::move(instances)),
      storage_(std::move(storage)), top_level_(top), destroy_(destroy) {}
TriangleScene::~TriangleScene() {
    if(top_level_ != VK_NULL_HANDLE) destroy_(device_, top_level_, nullptr);
}
TriangleScene::TriangleScene(TriangleScene&& other) noexcept { *this = std::move(other); }
TriangleScene& TriangleScene::operator=(TriangleScene&& other) noexcept {
    if(this == &other) return *this;
    std::swap(keep_,other.keep_); std::swap(device_,other.device_); std::swap(geometry_,other.geometry_);
    std::swap(instances_,other.instances_); std::swap(storage_,other.storage_);
    std::swap(top_level_,other.top_level_); std::swap(destroy_,other.destroy_); return *this;
}
std::shared_ptr<BottomLevel> Device::create_bottom_level(
    std::span<const float> positions, std::span<const std::uint32_t> indices,
    const TriangleSceneOptions& options) const {
    if(!impl_) throw std::logic_error("photara_vk Device is empty");
    if(!impl_->caps.enabled.ray_query || !impl_->caps.enabled.buffer_device_address)
        throw std::runtime_error("Device lacks hardware ray query support");
    if(positions.empty() || positions.size()%3 || indices.empty() || indices.size()%3 ||
        positions.size()/3 > UINT32_MAX || indices.size()/3 > UINT32_MAX)
        throw std::invalid_argument("Expected indexed float3 triangle geometry within 32-bit indexing");
    for(float value:positions) if(!std::isfinite(value)) throw std::invalid_argument("Non-finite ray-query position");
    for(auto index:indices) if(index >= positions.size()/3) throw std::out_of_range("Ray-query index out of bounds");
    auto fns=load_accel(impl_->device);
    auto usage=VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    auto vertices=make_buffer(*impl_,positions.size_bytes(),MemoryKind::host_visible,usage);
    auto index_buffer=make_buffer(*impl_,indices.size_bytes(),MemoryKind::host_visible,usage);
    vertices.upload(positions.data(),positions.size_bytes()); index_buffer.upload(indices.data(),indices.size_bytes());
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR; geometry.flags=options.geometry_flags;
    auto& triangles=geometry.geometry.triangles;
    triangles.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT; triangles.vertexStride=sizeof(float)*3;
    triangles.vertexData.deviceAddress=vertices.device_address(); triangles.maxVertex=std::uint32_t(positions.size()/3-1);
    triangles.indexType=VK_INDEX_TYPE_UINT32; triangles.indexData.deviceAddress=index_buffer.device_address();
    auto bottom=build_level(impl_,fns,VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,geometry,
        std::uint32_t(indices.size()/3),options.build_flags);
    VkAccelerationStructureDeviceAddressInfoKHR address_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address_info.accelerationStructure=bottom.handle;
    auto address=fns.address(impl_->device,&address_info);
    auto result=std::shared_ptr<BottomLevel>(new BottomLevel(impl_,impl_->device,std::move(vertices),std::move(index_buffer),
        std::move(bottom.storage),bottom.handle,address,fns.destroy));
    bottom.handle=VK_NULL_HANDLE;
    return result;
}
TriangleScene Device::create_instance_scene(std::span<const TriangleInstance> instances,
    VkBuildAccelerationStructureFlagsKHR flags) const {
    if(!impl_) throw std::logic_error("photara_vk Device is empty");
    if(instances.empty() || instances.size()>UINT32_MAX) throw std::invalid_argument("Expected nonempty TLAS instances");
    auto fns=load_accel(impl_->device);
    VkPhysicalDeviceAccelerationStructurePropertiesKHR props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; props2.pNext=&props;
    vkGetPhysicalDeviceProperties2(impl_->physical,&props2);
    if(instances.size()>props.maxInstanceCount) throw std::invalid_argument("TLAS instance count exceeds device limit");
    std::vector<VkAccelerationStructureInstanceKHR> records;
    std::vector<std::shared_ptr<const BottomLevel>> keep;
    for(const auto& instance:instances) {
        if(!instance.geometry || instance.geometry->device()!=impl_->device || instance.custom_index>0xffffff)
            throw std::invalid_argument("Invalid TLAS geometry/device/custom index");
        const auto& m=instance.transform.matrix;
        for(int r=0;r<3;++r) for(int c=0;c<4;++c) if(!std::isfinite(m[r][c])) throw std::invalid_argument("Non-finite instance transform");
        float determinant=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-
            m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
        if(std::abs(determinant)<1e-12f) throw std::invalid_argument("Singular instance transform");
        VkAccelerationStructureInstanceKHR record{};
        record.transform=instance.transform; record.instanceCustomIndex=instance.custom_index; record.mask=instance.mask;
        record.flags=instance.flags; record.accelerationStructureReference=instance.geometry->address();
        records.push_back(record); keep.push_back(instance.geometry);
    }
    auto input=make_buffer(*impl_,records.size()*sizeof(records[0]),MemoryKind::host_visible,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    input.upload(records.data(),records.size()*sizeof(records[0]));
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.data.deviceAddress=input.device_address();
    auto top=build_level(impl_,fns,VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,geometry,std::uint32_t(instances.size()),flags);
    auto result=TriangleScene(impl_,impl_->device,std::move(keep),std::move(input),std::move(top.storage),top.handle,fns.destroy);
    top.handle=VK_NULL_HANDLE; return result;
}
TriangleScene Device::create_triangle_scene(std::span<const float> positions,
    std::span<const std::uint32_t> indices,const TriangleSceneOptions& options) const {
    TriangleInstance instance; instance.geometry=create_bottom_level(positions,indices,options);
    return create_instance_scene(std::span(&instance,1),options.build_flags);
}
} // namespace photara::vk
