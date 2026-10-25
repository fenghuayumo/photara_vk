#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "photara_vk/accel.hpp"
#include "photara_vk/cache.hpp"
#include "photara_vk/graphics.hpp"
#include "photara_vk/query.hpp"

namespace photara::vk {

class ComputePipeline;
class CommandEncoder;

enum class BarrierPolicy : std::uint32_t {
    none = 0,
    after_compute = 1,
    after_compute_indirect = 2,
};

// Optional device capabilities. `DeviceRequest::want` asks the runtime to
// enable a feature when the hardware has it; `Capabilities::enabled` reports
// what actually came on.
struct Features {
    bool push_descriptors = false;
    bool buffer_atomic_f32 = false;
    // shaderBufferFloat64Atomics + shaderBufferFloat64AtomicAdd. Requires
    // shader_float64, which create() turns on when this request is granted.
    bool buffer_atomic_f64 = false;
    bool shader_float64 = false;
    bool shader_int64 = false;
    bool storage_image_write_without_format = false;
    bool geometry_shader = false;
    bool tessellation_shader = false;
    bool synchronization2 = false;
    bool integer_dot_product = false;
    bool buffer_device_address = false;
    bool ray_query = false;
    bool ray_tracing_pipeline = false;
    bool external_memory_win32 = false;
    bool external_semaphore_win32 = false;
    bool timeline_semaphore = false;
    bool graphics = false;
    bool present = false;
    bool sampled_image_nonuniform = false;
    bool shader_device_clock = false;
};

inline constexpr std::uint32_t k_best_device =
    std::numeric_limits<std::uint32_t>::max();

struct DeviceRequest {
    std::uint32_t api_version = VK_API_VERSION_1_2;
    // Physical-device index in vkEnumeratePhysicalDevices order, or
    // k_best_device to pick the highest-scoring compute GPU.
    std::uint32_t device_index = k_best_device;
    bool validation = false;
    bool synchronization_validation = false;
    Features want{.push_descriptors = true};
    std::vector<const char *> instance_extensions;
    std::vector<const char *> device_extensions;
    // Called after the physical device is selected and before the logical
    // device is created. Optional SDKs (NGX/DLSS) use it to append the device
    // extensions they require without teaching this RHI about the SDK.
    std::function<void(VkInstance, VkPhysicalDevice, std::vector<const char *> &)>
        device_extension_query;
    VkSurfaceKHR present_surface = VK_NULL_HANDLE;
};

struct ExternalDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = VK_QUEUE_FAMILY_IGNORED;
    Features enabled{};

    [[nodiscard]] bool valid() const noexcept {
        return device != VK_NULL_HANDLE && queue != VK_NULL_HANDLE &&
               physical != VK_NULL_HANDLE;
    }
};

struct Capabilities {
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t api_version = 0;
    std::uint32_t subgroup_size = 0;
    std::uint32_t max_shared_bytes = 0;
    std::uint32_t max_push_constants = 0;
    std::uint32_t max_storage_buffer_range = 0;
    std::uint32_t timestamp_valid_bits = 0;
    double timestamp_period_nanoseconds = 0.0;
    Features enabled{};
};

class Device {
public:
    struct Impl;

    Device() = default;
    Device(const Device &) = default;
    Device &operator=(const Device &) = default;
    Device(Device &&) noexcept = default;
    Device &operator=(Device &&) noexcept = default;
    ~Device() = default;

    [[nodiscard]] static Device create(const DeviceRequest &request = {});
    [[nodiscard]] static Device adopt(const ExternalDevice &external);
    [[nodiscard]] static std::vector<Capabilities> enumerate(
        const DeviceRequest &request = {});

    [[nodiscard]] bool valid() const noexcept { return impl_ != nullptr; }
    [[nodiscard]] VkInstance instance() const noexcept;
    [[nodiscard]] VkPhysicalDevice physical() const noexcept;
    [[nodiscard]] VkDevice handle() const noexcept;
    [[nodiscard]] VkQueue queue() const noexcept;
    [[nodiscard]] std::uint32_t queue_family() const noexcept;
    [[nodiscard]] const Capabilities &caps() const;
    [[nodiscard]] ExternalDevice handles() const;
    [[nodiscard]] bool owns_device() const noexcept;
    [[nodiscard]] std::uint32_t validation_error_count() const noexcept;
    void reset_validation_errors() const noexcept;
    [[nodiscard]] std::mutex &queue_mutex() const;

    [[nodiscard]] Buffer create_buffer(VkDeviceSize bytes, MemoryKind kind,
                                       VkBufferUsageFlags extra_usage = 0) const;
    [[nodiscard]] TimestampPool create_timestamp_pool(
        std::uint32_t timestamp_count) const;
    [[nodiscard]] PipelineCache create_pipeline_cache(
        std::span<const std::byte> initial_data = {}) const;
    [[nodiscard]] ComputePipeline create_compute(
        std::span<const std::byte> spirv, std::uint32_t storage_buffer_count,
        std::uint32_t push_bytes, const PipelineCache *cache = nullptr) const;
    [[nodiscard]] ComputePipeline create_compute(
        std::span<const std::byte> spirv,
        std::span<const VkDescriptorType> descriptor_types, std::uint32_t push_bytes,
        const PipelineCache *cache = nullptr,
        std::span<const std::uint32_t> descriptor_counts = {}) const;
    [[nodiscard]] CommandEncoder encoder(
        BarrierPolicy policy = BarrierPolicy::after_compute) const;
    [[nodiscard]] TriangleScene create_triangle_scene(
        std::span<const float> positions, std::span<const std::uint32_t> indices,
        const TriangleSceneOptions &options = {}) const;
    [[nodiscard]] std::shared_ptr<BottomLevel> create_bottom_level(
        std::span<const float> positions, std::span<const std::uint32_t> indices,
        const TriangleSceneOptions &options = {}) const;
    [[nodiscard]] TriangleScene create_instance_scene(
        std::span<const TriangleInstance> instances,
        VkBuildAccelerationStructureFlagsKHR flags =
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR) const;
    void update_instance_scene(TriangleScene &scene, std::span<const TriangleInstance> instances) const;
    [[nodiscard]] Image create_image(const ImageDesc &desc) const;
    [[nodiscard]] Sampler create_sampler(bool repeat = false, float max_lod = 0) const;
    [[nodiscard]] bool supports_linear_sampling(VkFormat format) const;
    [[nodiscard]] ImageView create_image_view(VkImage image, VkFormat format,
                                             VkImageAspectFlags aspect, std::uint32_t mip_levels = 1) const;
    [[nodiscard]] RenderPass create_render_pass(
        std::span<const AttachmentDesc> attachments,
        std::span<const SubpassDependencyDesc> dependencies) const;
    [[nodiscard]] Framebuffer create_framebuffer(
        const RenderPass &render_pass, std::span<const VkImageView> attachments,
        std::uint32_t width, std::uint32_t height) const;
    [[nodiscard]] GraphicsPipeline create_graphics(const GraphicsDesc &desc) const;

    void wait_idle() const;

private:
    std::shared_ptr<Impl> impl_;
};

// One mutex per VkQueue for the whole process. Every Device on that queue,
// and every raw submit that takes QueueLock, shares it. Entries are never
// removed, so the mutex address stays stable across map growth.
[[nodiscard]] std::mutex &queue_mutex(VkQueue queue);

class QueueLock {
public:
    explicit QueueLock(VkQueue queue);
    explicit QueueLock(const Device &device);
    QueueLock(const QueueLock &) = delete;
    QueueLock &operator=(const QueueLock &) = delete;

private:
    std::unique_lock<std::mutex> lock_;
};

}  // namespace photara::vk
