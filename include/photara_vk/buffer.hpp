#pragma once

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan.h>

namespace photara::vk {

enum class MemoryKind : std::uint32_t {
    device_local = 0,
    host_visible = 1,
    host_cached = 2,
    host_visible_device_local = 3,
};

class Buffer {
public:
    Buffer() = default;
    Buffer(VkPhysicalDevice physical, VkDevice logical, VkDeviceSize bytes,
           MemoryKind kind, VkBufferUsageFlags usage, bool device_address);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] VkBuffer handle() const noexcept { return handle_; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }
    [[nodiscard]] VkDeviceSize offset() const noexcept { return offset_; }
    [[nodiscard]] void* mapped() const noexcept { return mapped_; }
    [[nodiscard]] bool coherent() const noexcept { return coherent_; }
    [[nodiscard]] bool host_visible() const noexcept { return mapped_ != nullptr; }

    void upload(const void* source, std::size_t bytes, std::size_t offset = 0);
    void download(void* destination, std::size_t bytes, std::size_t offset = 0) const;
    void flush(VkDeviceSize offset, VkDeviceSize bytes) const;
    void invalidate(VkDeviceSize offset, VkDeviceSize bytes) const;
    [[nodiscard]] VkDeviceAddress device_address() const;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer handle_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkDeviceSize offset_ = 0;
    void* mapped_ = nullptr;
    bool coherent_ = true;
    VkDeviceSize non_coherent_atom_size_ = 1;
    bool owns_ = true;
};

}  // namespace photara::vk
