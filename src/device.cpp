#include "device_impl.hpp"

#include "photara_vk/check.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace photara::vk {
namespace {

bool has_instance_layer(const char* name) {
    std::uint32_t count = 0;
    check_vk(vkEnumerateInstanceLayerProperties(&count, nullptr),
             "vkEnumerateInstanceLayerProperties");
    std::vector<VkLayerProperties> properties(count);
    check_vk(vkEnumerateInstanceLayerProperties(&count, properties.data()),
             "vkEnumerateInstanceLayerProperties");
    return std::any_of(properties.begin(), properties.end(),
                       [name](const VkLayerProperties& property) {
                           return std::strcmp(property.layerName, name) == 0;
                       });
}

bool has_instance_extension(const char* name) {
    std::uint32_t count = 0;
    check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
             "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> properties(count);
    check_vk(
        vkEnumerateInstanceExtensionProperties(nullptr, &count, properties.data()),
        "vkEnumerateInstanceExtensionProperties");
    return std::any_of(properties.begin(), properties.end(),
                       [name](const VkExtensionProperties& property) {
                           return std::strcmp(property.extensionName, name) == 0;
                       });
}

bool has_device_extension(VkPhysicalDevice physical, const char* name) {
    std::uint32_t count = 0;
    check_vk(
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> properties(count);
    check_vk(vkEnumerateDeviceExtensionProperties(
                 physical, nullptr, &count, properties.data()),
             "vkEnumerateDeviceExtensionProperties");
    return std::any_of(properties.begin(), properties.end(),
                       [name](const VkExtensionProperties& property) {
                           return std::strcmp(property.extensionName, name) == 0;
                       });
}

void add_unique(std::vector<const char*>& list, const char* name) {
    for (const char* existing : list) {
        if (std::strcmp(existing, name) == 0) return;
    }
    list.push_back(name);
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (data != nullptr && data->pMessage != nullptr) {
        std::fprintf(stderr, "photara_vk: %s\n", data->pMessage);
    }
    return VK_FALSE;
}

std::uint32_t find_queue_family(
    VkPhysicalDevice physical, const bool graphics, const bool present,
    VkSurfaceKHR surface) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto flags = families[index].queueFlags;
        if ((flags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
        if (graphics && (flags & VK_QUEUE_GRAPHICS_BIT) == 0) continue;
        if (present) {
            if (surface == VK_NULL_HANDLE) {
                throw std::invalid_argument(
                    "DeviceRequest.want.present requires present_surface");
            }
            VkBool32 supported = VK_FALSE;
            check_vk(vkGetPhysicalDeviceSurfaceSupportKHR(
                         physical, index, surface, &supported),
                     "vkGetPhysicalDeviceSurfaceSupportKHR");
            if (supported != VK_TRUE) continue;
        }
        return index;
    }
    throw std::runtime_error("No Vulkan queue family matches the device request");
}

int device_score(VkPhysicalDevice physical) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    try {
        (void)find_queue_family(physical, false, false, VK_NULL_HANDLE);
    } catch (...) {
        return -1;
    }
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return 300;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return 100;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) return -1;
    return 0;
}

Capabilities make_caps(VkPhysicalDevice physical) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    Capabilities caps;
    caps.name = properties.deviceName;
    caps.vendor_id = properties.vendorID;
    caps.device_id = properties.deviceID;
    caps.api_version = properties.apiVersion;
    caps.max_shared_bytes = properties.limits.maxComputeSharedMemorySize;
    caps.max_push_constants = properties.limits.maxPushConstantsSize;
    VkPhysicalDeviceSubgroupProperties subgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical, &properties2);
    caps.subgroup_size = subgroup.subgroupSize;
    return caps;
}

void create_instance(Device::Impl& impl, const DeviceRequest& request) {
    std::vector<const char*> layers;
    std::vector<const char*> extensions = request.instance_extensions;
    if (request.validation) {
        if (!has_instance_layer("VK_LAYER_KHRONOS_validation")) {
            throw std::runtime_error(
                "Vulkan validation was requested but VK_LAYER_KHRONOS_validation "
                "is unavailable");
        }
        layers.push_back("VK_LAYER_KHRONOS_validation");
        add_unique(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    const bool portability =
        has_instance_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    if (portability) {
        add_unique(extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "photara_vk";
    application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.pEngineName = "photara_vk";
    application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.apiVersion = request.api_version;

    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
#ifdef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    if (portability) {
        create.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif
    create.pApplicationInfo = &application;
    create.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    create.ppEnabledLayerNames = layers.data();
    create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    check_vk(vkCreateInstance(&create, nullptr, &impl.instance), "vkCreateInstance");
    impl.owns_instance = true;

    if (request.validation) {
        const auto create_messenger =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(impl.instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create_messenger != nullptr) {
            VkDebugUtilsMessengerCreateInfoEXT debug{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debug.pfnUserCallback = debug_callback;
            check_vk(create_messenger(impl.instance, &debug, nullptr,
                                      &impl.debug_messenger),
                     "vkCreateDebugUtilsMessengerEXT");
        }
    }
}

void pick_physical(Device::Impl& impl, const DeviceRequest& request) {
    std::uint32_t count = 0;
    check_vk(vkEnumeratePhysicalDevices(impl.instance, &count, nullptr),
             "vkEnumeratePhysicalDevices");
    if (count == 0) throw std::runtime_error("No Vulkan devices were found");
    std::vector<VkPhysicalDevice> devices(count);
    check_vk(vkEnumeratePhysicalDevices(impl.instance, &count, devices.data()),
             "vkEnumeratePhysicalDevices");

    if (request.device_index == k_best_device) {
        int best = -1;
        for (VkPhysicalDevice candidate : devices) {
            const int score = device_score(candidate);
            if (score > best) {
                best = score;
                impl.physical = candidate;
            }
        }
        if (impl.physical == VK_NULL_HANDLE) {
            throw std::runtime_error("No Vulkan compute device was found");
        }
    } else {
        if (request.device_index >= count) {
            throw std::out_of_range("Vulkan device_index is out of range");
        }
        impl.physical = devices[request.device_index];
    }

    impl.caps = make_caps(impl.physical);
    impl.queue_family = find_queue_family(
        impl.physical, request.want.graphics, request.want.present,
        request.present_surface);
    vkGetPhysicalDeviceMemoryProperties(impl.physical, &impl.memory);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(impl.physical, &properties);
    impl.non_coherent_atom_size =
        std::max<VkDeviceSize>(properties.limits.nonCoherentAtomSize, 1);
}

void create_logical(Device::Impl& impl, const DeviceRequest& request) {
    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = impl.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    std::vector<const char*> extensions;
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
    if (has_device_extension(impl.physical, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#endif

    Features& enabled = impl.caps.enabled;
    enabled = {};

    VkPhysicalDeviceVulkan12Features vk12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan12Features vk12_query{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 features_query{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features_query.pNext = &vk12_query;
    vkGetPhysicalDeviceFeatures2(impl.physical, &features_query);

#ifdef VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME
    if (request.want.push_descriptors &&
        has_device_extension(impl.physical, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        enabled.push_descriptors = true;
    }
#endif

    if (request.want.buffer_device_address && vk12_query.bufferDeviceAddress) {
        vk12.bufferDeviceAddress = VK_TRUE;
        enabled.buffer_device_address = true;
    }
    if (request.want.timeline_semaphore && vk12_query.timelineSemaphore) {
        vk12.timelineSemaphore = VK_TRUE;
        enabled.timeline_semaphore = true;
    }

#ifdef VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    if (request.want.buffer_atomic_f32 &&
        has_device_extension(impl.physical, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_query{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        query.pNext = &atomic_query;
        vkGetPhysicalDeviceFeatures2(impl.physical, &query);
        if (atomic_query.shaderBufferFloat32AtomicAdd) {
            extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
            atomic.shaderBufferFloat32AtomicAdd = VK_TRUE;
            enabled.buffer_atomic_f32 = true;
        }
    }
#endif

    VkPhysicalDeviceShaderIntegerDotProductFeatures dot{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    if (request.want.integer_dot_product) {
        const bool has_ext = has_device_extension(
            impl.physical, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(impl.physical, &properties);
        if (has_ext || properties.apiVersion >= VK_API_VERSION_1_3) {
            VkPhysicalDeviceShaderIntegerDotProductFeatures dot_query{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            query.pNext = &dot_query;
            vkGetPhysicalDeviceFeatures2(impl.physical, &query);
            if (dot_query.shaderIntegerDotProduct) {
                if (has_ext) {
                    extensions.push_back(
                        VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
                }
                dot.shaderIntegerDotProduct = VK_TRUE;
                enabled.integer_dot_product = true;
            }
        }
    }

    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_pipeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    const bool want_rt =
        request.want.ray_query || request.want.ray_tracing_pipeline;
    if (want_rt) {
        if (!enabled.buffer_device_address && vk12_query.bufferDeviceAddress) {
            vk12.bufferDeviceAddress = VK_TRUE;
            enabled.buffer_device_address = true;
        }
        const bool has_as = has_device_extension(
            impl.physical, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        const bool has_deferred = has_device_extension(
            impl.physical, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        if (has_as && has_deferred && enabled.buffer_device_address) {
            VkPhysicalDeviceAccelerationStructureFeaturesKHR as_query{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            query.pNext = &as_query;
            vkGetPhysicalDeviceFeatures2(impl.physical, &query);
            if (as_query.accelerationStructure) {
                extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
                extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
                acceleration.accelerationStructure = VK_TRUE;
                if (request.want.ray_query &&
                    has_device_extension(impl.physical, VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
                    VkPhysicalDeviceRayQueryFeaturesKHR rq_query{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
                    VkPhysicalDeviceFeatures2 rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                    rq.pNext = &rq_query;
                    vkGetPhysicalDeviceFeatures2(impl.physical, &rq);
                    if (rq_query.rayQuery) {
                        extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
                        ray_query.rayQuery = VK_TRUE;
                        enabled.ray_query = true;
                    }
                }
                if (request.want.ray_tracing_pipeline &&
                    has_device_extension(
                        impl.physical, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
                    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtp_query{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
                    VkPhysicalDeviceFeatures2 rtp{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                    rtp.pNext = &rtp_query;
                    vkGetPhysicalDeviceFeatures2(impl.physical, &rtp);
                    if (rtp_query.rayTracingPipeline) {
                        extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
                        ray_pipeline.rayTracingPipeline = VK_TRUE;
                        enabled.ray_tracing_pipeline = true;
                    }
                }
            }
        }
    }

#if defined(_WIN32)
    if (request.want.external_memory_win32) {
        const char* names[] = {
            "VK_KHR_external_memory",
            "VK_KHR_external_memory_win32",
        };
        if (has_device_extension(impl.physical, names[0]) &&
            has_device_extension(impl.physical, names[1])) {
            extensions.push_back(names[0]);
            extensions.push_back(names[1]);
            enabled.external_memory_win32 = true;
        }
    }
    if (request.want.external_semaphore_win32) {
        const char* names[] = {
            "VK_KHR_external_semaphore",
            "VK_KHR_external_semaphore_win32",
        };
        if (has_device_extension(impl.physical, names[0]) &&
            has_device_extension(impl.physical, names[1])) {
            extensions.push_back(names[0]);
            extensions.push_back(names[1]);
            enabled.external_semaphore_win32 = true;
        }
    }
#endif

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(impl.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        impl.physical, &family_count, families.data());
    if (impl.queue_family < family_count) {
        enabled.graphics =
            (families[impl.queue_family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
    }
    enabled.present = request.want.present;

    void* pNext = nullptr;
    auto push_feature = [&](auto& feature) {
        feature.pNext = pNext;
        pNext = &feature;
    };
    if (vk12.bufferDeviceAddress || vk12.timelineSemaphore) push_feature(vk12);
#ifdef VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
    if (atomic.shaderBufferFloat32AtomicAdd) push_feature(atomic);
#endif
    if (dot.shaderIntegerDotProduct) push_feature(dot);
    if (acceleration.accelerationStructure) push_feature(acceleration);
    if (ray_query.rayQuery) push_feature(ray_query);
    if (ray_pipeline.rayTracingPipeline) push_feature(ray_pipeline);

    VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    create.pNext = pNext;
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = &queue_info;
    create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    check_vk(vkCreateDevice(impl.physical, &create, nullptr, &impl.device),
             "vkCreateDevice");
    impl.owns_device = true;
    vkGetDeviceQueue(impl.device, impl.queue_family, 0, &impl.queue);

#ifdef VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME
    if (enabled.push_descriptors) {
        impl.cmd_push_descriptor = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetDeviceProcAddr(impl.device, "vkCmdPushDescriptorSetKHR"));
        if (impl.cmd_push_descriptor == nullptr) {
            enabled.push_descriptors = false;
        }
    }
#endif
}

void create_pools(Device::Impl& impl) {
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = impl.queue_family;
    check_vk(vkCreateCommandPool(impl.device, &pool, nullptr, &impl.command_pool),
             "vkCreateCommandPool");
}

}  // namespace

Device::Impl::~Impl() {
    if (device != VK_NULL_HANDLE) {
        std::lock_guard lock(queue_mutex);
        vkDeviceWaitIdle(device);
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
        }
        if (owns_device) {
            vkDestroyDevice(device, nullptr);
        }
    }
    if (debug_messenger != VK_NULL_HANDLE && instance != VK_NULL_HANDLE) {
        const auto destroy_messenger =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_messenger != nullptr) {
            destroy_messenger(instance, debug_messenger, nullptr);
        }
    }
    if (owns_instance && instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}

Device Device::create(const DeviceRequest& request) {
    Device device;
    device.impl_ = std::make_shared<Impl>();
    create_instance(*device.impl_, request);
    pick_physical(*device.impl_, request);
    create_logical(*device.impl_, request);
    create_pools(*device.impl_);
    return device;
}

Device Device::adopt(const ExternalDevice& external) {
    if (!external.valid()) {
        throw std::invalid_argument(
            "Adopting a Vulkan device requires physical, device, and queue");
    }
    try {
        (void)find_queue_family(external.physical, false, false, VK_NULL_HANDLE);
    } catch (...) {
        throw std::invalid_argument(
            "Adopting a Vulkan device requires a compute-capable physical device");
    }
    Device device;
    device.impl_ = std::make_shared<Impl>();
    Impl& impl = *device.impl_;
    impl.instance = external.instance;
    impl.physical = external.physical;
    impl.device = external.device;
    impl.queue = external.queue;
    impl.queue_family = external.queue_family;
    impl.caps = make_caps(external.physical);
    impl.caps.enabled = external.enabled;
    vkGetPhysicalDeviceMemoryProperties(impl.physical, &impl.memory);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(impl.physical, &properties);
    impl.non_coherent_atom_size =
        std::max<VkDeviceSize>(properties.limits.nonCoherentAtomSize, 1);
    if (impl.caps.enabled.push_descriptors) {
        impl.cmd_push_descriptor = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetDeviceProcAddr(impl.device, "vkCmdPushDescriptorSetKHR"));
        if (impl.cmd_push_descriptor == nullptr) {
            impl.caps.enabled.push_descriptors = false;
        }
    }
    create_pools(impl);
    return device;
}

std::vector<Capabilities> Device::enumerate(const DeviceRequest& request) {
    DeviceRequest listing = request;
    listing.want = {};
    listing.present_surface = VK_NULL_HANDLE;
    Device::Impl impl;
    create_instance(impl, listing);
    std::uint32_t count = 0;
    check_vk(vkEnumeratePhysicalDevices(impl.instance, &count, nullptr),
             "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> devices(count);
    if (count > 0) {
        check_vk(vkEnumeratePhysicalDevices(impl.instance, &count, devices.data()),
                 "vkEnumeratePhysicalDevices");
    }
    std::vector<Capabilities> result;
    result.reserve(devices.size());
    for (VkPhysicalDevice physical : devices) {
        result.push_back(make_caps(physical));
    }
    return result;
}

VkInstance Device::instance() const noexcept {
    return impl_ ? impl_->instance : VK_NULL_HANDLE;
}

VkPhysicalDevice Device::physical() const noexcept {
    return impl_ ? impl_->physical : VK_NULL_HANDLE;
}

VkDevice Device::handle() const noexcept {
    return impl_ ? impl_->device : VK_NULL_HANDLE;
}

VkQueue Device::queue() const noexcept {
    return impl_ ? impl_->queue : VK_NULL_HANDLE;
}

std::uint32_t Device::queue_family() const noexcept {
    return impl_ ? impl_->queue_family : VK_QUEUE_FAMILY_IGNORED;
}

const Capabilities& Device::caps() const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return impl_->caps;
}

ExternalDevice Device::handles() const {
    ExternalDevice result;
    if (!impl_) return result;
    result.instance = impl_->instance;
    result.physical = impl_->physical;
    result.device = impl_->device;
    result.queue = impl_->queue;
    result.queue_family = impl_->queue_family;
    result.enabled = impl_->caps.enabled;
    return result;
}

bool Device::owns_device() const noexcept {
    return impl_ != nullptr && impl_->owns_device;
}

std::mutex& Device::queue_mutex() const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return impl_->queue_mutex;
}

void Device::wait_idle() const {
    if (!impl_ || impl_->device == VK_NULL_HANDLE) return;
    std::lock_guard lock(impl_->queue_mutex);
    check_vk(vkDeviceWaitIdle(impl_->device), "vkDeviceWaitIdle");
}

}  // namespace photara::vk
