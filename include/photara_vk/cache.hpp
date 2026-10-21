#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace photara::vk {

// Opaque driver data: persist only bytes returned by data(). Device creation
// validates the Vulkan header against the selected GPU/driver cache UUID.
// Serialization and pipeline creation using this cache are synchronized.
class PipelineCache {
public:
    PipelineCache() = default;
    ~PipelineCache() = default;
    PipelineCache(PipelineCache&&) noexcept = default;
    PipelineCache& operator=(PipelineCache&&) noexcept = default;
    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;
    [[nodiscard]] bool valid() const noexcept { return impl_ != nullptr; }
    [[nodiscard]] std::vector<std::byte> data() const;

private:
    friend class Device;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace photara::vk
