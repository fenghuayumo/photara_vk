#include "photara_vk/buffer.hpp"

#include "device_impl.hpp"
#include "photara_vk/check.hpp"

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>

namespace photara::vk {
namespace {

std::uint32_t find_memory_type(
    VkPhysicalDevice physical, std::uint32_t type_bits,
    VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((type_bits & (1U << index)) != 0 &&
            (properties.memoryTypes[index].propertyFlags & required) == required) {
            return index;
        }
    }
    throw std::runtime_error("No compatible Vulkan memory type was found");
}

std::optional<std::uint32_t> try_find_memory_type(
    VkPhysicalDevice physical, std::uint32_t type_bits,
    VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((type_bits & (1U << index)) != 0 &&
            (properties.memoryTypes[index].propertyFlags & required) == required) {
            return index;
        }
    }
    return std::nullopt;
}

std::uint32_t select_memory_type(
    VkPhysicalDevice physical, std::uint32_t type_bits, MemoryKind kind) {
    const VkMemoryPropertyFlags host_coherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    switch (kind) {
    case MemoryKind::host_cached:
        return try_find_memory_type(
                   physical, type_bits,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            .value_or(try_find_memory_type(
                          physical, type_bits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                          .value_or(find_memory_type(physical, type_bits, host_coherent)));
    case MemoryKind::host_visible_device_local:
        return try_find_memory_type(
                   physical, type_bits,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            .value_or(find_memory_type(physical, type_bits, host_coherent));
    case MemoryKind::host_visible:
        return find_memory_type(physical, type_bits, host_coherent);
    case MemoryKind::device_local:
    default:
        return try_find_memory_type(
                   physical, type_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            .value_or(find_memory_type(physical, type_bits, host_coherent));
    }
}

}  // namespace

Buffer::Buffer(
    VkPhysicalDevice physical, VkDevice logical, VkDeviceSize bytes,
    MemoryKind kind, VkBufferUsageFlags usage, const bool device_address)
    : device_(logical), size_(std::max<VkDeviceSize>(bytes, 4)) {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size_;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check_vk(vkCreateBuffer(device_, &buffer_info, nullptr, &handle_),
             "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, handle_, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex =
        select_memory_type(physical, requirements.memoryTypeBits, kind);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if (device_address) {
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        allocation.pNext = &flags;
    }
    check_vk(vkAllocateMemory(device_, &allocation, nullptr, &memory_),
             "vkAllocateMemory");
    check_vk(vkBindBufferMemory(device_, handle_, memory_, 0), "vkBindBufferMemory");

    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    const auto memory_flags =
        properties.memoryTypes[allocation.memoryTypeIndex].propertyFlags;
    if ((memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        check_vk(vkMapMemory(device_, memory_, 0, size_, 0, &mapped_), "vkMapMemory");
        coherent_ = (memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkPhysicalDeviceProperties limits{};
        vkGetPhysicalDeviceProperties(physical, &limits);
        non_coherent_atom_size_ =
            std::max<VkDeviceSize>(limits.limits.nonCoherentAtomSize, 1);
    }
}

Buffer::~Buffer() {
    if (device_ == VK_NULL_HANDLE || !owns_) return;
    if (mapped_ != nullptr) vkUnmapMemory(device_, memory_);
    if (handle_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, handle_, nullptr);
    if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
}

Buffer::Buffer(Buffer&& other) noexcept {
    *this = std::move(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this == &other) return *this;
    std::swap(device_, other.device_);
    std::swap(handle_, other.handle_);
    std::swap(memory_, other.memory_);
    std::swap(size_, other.size_);
    std::swap(offset_, other.offset_);
    std::swap(mapped_, other.mapped_);
    std::swap(coherent_, other.coherent_);
    std::swap(non_coherent_atom_size_, other.non_coherent_atom_size_);
    std::swap(owns_, other.owns_);
    return *this;
}

void Buffer::upload(
    const void* source, const std::size_t bytes, const std::size_t offset) {
    if (mapped_ == nullptr) {
        throw std::logic_error(
            "Buffer is device-local; upload through CommandEncoder::copy");
    }
    if (offset + bytes > size_) {
        throw std::out_of_range("Buffer upload exceeds allocation");
    }
    if (bytes == 0) return;
    std::memcpy(static_cast<std::byte*>(mapped_) + offset, source, bytes);
    flush(static_cast<VkDeviceSize>(offset), static_cast<VkDeviceSize>(bytes));
}

void Buffer::download(
    void* destination, const std::size_t bytes, const std::size_t offset) const {
    if (mapped_ == nullptr) {
        throw std::logic_error(
            "Buffer is device-local; download through CommandEncoder::copy");
    }
    if (offset + bytes > size_) {
        throw std::out_of_range("Buffer download exceeds allocation");
    }
    if (bytes == 0) return;
    invalidate(static_cast<VkDeviceSize>(offset), static_cast<VkDeviceSize>(bytes));
    std::memcpy(destination, static_cast<const std::byte*>(mapped_) + offset, bytes);
}

void Buffer::flush(const VkDeviceSize offset, const VkDeviceSize bytes) const {
    if (mapped_ == nullptr || coherent_ || bytes == 0) return;
    const VkDeviceSize atom = non_coherent_atom_size_;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory_;
    range.offset = (offset / atom) * atom;
    const VkDeviceSize end = offset + bytes;
    const VkDeviceSize aligned_end = std::min(size_, ((end + atom - 1) / atom) * atom);
    range.size = aligned_end - range.offset;
    check_vk(vkFlushMappedMemoryRanges(device_, 1, &range),
             "vkFlushMappedMemoryRanges");
}

void Buffer::invalidate(const VkDeviceSize offset, const VkDeviceSize bytes) const {
    if (mapped_ == nullptr || coherent_ || bytes == 0) return;
    const VkDeviceSize atom = non_coherent_atom_size_;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory_;
    range.offset = (offset / atom) * atom;
    const VkDeviceSize end = offset + bytes;
    const VkDeviceSize aligned_end = std::min(size_, ((end + atom - 1) / atom) * atom);
    range.size = aligned_end - range.offset;
    check_vk(vkInvalidateMappedMemoryRanges(device_, 1, &range),
             "vkInvalidateMappedMemoryRanges");
}

VkDeviceAddress Buffer::device_address() const {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = handle_;
    return vkGetBufferDeviceAddress(device_, &info);
}

Buffer Device::create_buffer(
    const VkDeviceSize bytes, const MemoryKind kind,
    const VkBufferUsageFlags extra_usage) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra_usage;
    const bool device_address = impl_->caps.enabled.buffer_device_address;
    if (device_address) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    return Buffer(impl_->physical, impl_->device, bytes, kind, usage, device_address);
}

}  // namespace photara::vk
