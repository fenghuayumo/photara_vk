#include "photara_vk/cache.hpp"

#include <cstring>
#include <stdexcept>

#include "cache_impl.hpp"
#include "device_impl.hpp"
#include "photara_vk/check.hpp"

namespace photara::vk {

PipelineCache Device::create_pipeline_cache(
    std::span<const std::byte> initial_data) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    if (!initial_data.empty()) {
        VkPipelineCacheHeaderVersionOne header{};
        if (initial_data.size() < sizeof(header))
            throw std::invalid_argument("Truncated pipeline cache header");
        std::memcpy(&header, initial_data.data(), sizeof(header));
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(impl_->physical, &properties);
        if (header.headerSize != sizeof(header) ||
            header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
            header.vendorID != properties.vendorID ||
            header.deviceID != properties.deviceID ||
            std::memcmp(header.pipelineCacheUUID, properties.pipelineCacheUUID,
                        VK_UUID_SIZE) != 0)
            throw std::invalid_argument(
                "Pipeline cache is incompatible with this GPU/driver");
    }
    PipelineCache cache;
    cache.impl_ = std::make_shared<PipelineCache::Impl>();
    cache.impl_->keep = impl_;
    cache.impl_->device = impl_->device;
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = initial_data.size();
    info.pInitialData = initial_data.empty() ? nullptr : initial_data.data();
    check_vk(vkCreatePipelineCache(impl_->device, &info, nullptr, &cache.impl_->handle),
             "vkCreatePipelineCache");
    return cache;
}

std::vector<std::byte> PipelineCache::data() const {
    if (!impl_) throw std::logic_error("PipelineCache is empty");
    std::lock_guard lock(impl_->mutex);
    for (;;) {
        std::size_t size = 0;
        check_vk(vkGetPipelineCacheData(impl_->device, impl_->handle, &size, nullptr),
                 "vkGetPipelineCacheData size");
        std::vector<std::byte> bytes(size);
        const VkResult result =
            vkGetPipelineCacheData(impl_->device, impl_->handle, &size, bytes.data());
        if (result == VK_INCOMPLETE) continue;
        check_vk(result, "vkGetPipelineCacheData");
        bytes.resize(size);
        return bytes;
    }
}
}  // namespace photara::vk
