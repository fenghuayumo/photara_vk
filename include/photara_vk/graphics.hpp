#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <vulkan/vulkan.h>

namespace photara::vk {

struct ImageDesc {
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageUsageFlags usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkMemoryPropertyFlags memory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
};

class Image {
public:
    Image() = default;
    Image(VkPhysicalDevice physical, VkDevice device, const ImageDesc& desc);
    ~Image();
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    [[nodiscard]] bool valid() const noexcept { return handle_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkImage handle() const noexcept { return handle_; }
    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }
    [[nodiscard]] std::uint32_t height() const noexcept { return height_; }
    [[nodiscard]] VkFormat format() const noexcept { return format_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkImage handle_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
};

class ImageView {
public:
    ImageView() = default;
    ImageView(
        VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspect);
    ~ImageView();
    ImageView(ImageView&& other) noexcept;
    ImageView& operator=(ImageView&& other) noexcept;
    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    [[nodiscard]] bool valid() const noexcept { return handle_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkImageView handle() const noexcept { return handle_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkImageView handle_ = VK_NULL_HANDLE;
};

struct AttachmentDesc {
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkAttachmentLoadOp load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkAttachmentStoreOp store_op = VK_ATTACHMENT_STORE_OP_STORE;
    VkAttachmentLoadOp stencil_load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    VkAttachmentStoreOp stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout final_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bool depth = false;
};

struct SubpassDependencyDesc {
    std::uint32_t src_subpass = VK_SUBPASS_EXTERNAL;
    std::uint32_t dst_subpass = 0;
    VkPipelineStageFlags src_stage = 0;
    VkPipelineStageFlags dst_stage = 0;
    VkAccessFlags src_access = 0;
    VkAccessFlags dst_access = 0;
};

class RenderPass {
public:
    RenderPass() = default;
    RenderPass(
        VkDevice device, std::span<const AttachmentDesc> attachments,
        std::span<const SubpassDependencyDesc> dependencies);
    ~RenderPass();
    RenderPass(RenderPass&& other) noexcept;
    RenderPass& operator=(RenderPass&& other) noexcept;
    RenderPass(const RenderPass&) = delete;
    RenderPass& operator=(const RenderPass&) = delete;

    [[nodiscard]] bool valid() const noexcept { return handle_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkRenderPass handle() const noexcept { return handle_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkRenderPass handle_ = VK_NULL_HANDLE;
};

class Framebuffer {
public:
    Framebuffer() = default;
    Framebuffer(
        VkDevice device, VkRenderPass render_pass,
        std::span<const VkImageView> attachments, std::uint32_t width,
        std::uint32_t height);
    ~Framebuffer();
    Framebuffer(Framebuffer&& other) noexcept;
    Framebuffer& operator=(Framebuffer&& other) noexcept;
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    [[nodiscard]] bool valid() const noexcept { return handle_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkFramebuffer handle() const noexcept { return handle_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkFramebuffer handle_ = VK_NULL_HANDLE;
};

struct VertexAttribute {
    std::uint32_t location = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t offset = 0;
};

struct ColorBlend {
    bool enable = false;
    VkBlendFactor src_color = VK_BLEND_FACTOR_ONE;
    VkBlendFactor dst_color = VK_BLEND_FACTOR_ZERO;
    VkBlendOp color_op = VK_BLEND_OP_ADD;
    VkBlendFactor src_alpha = VK_BLEND_FACTOR_ONE;
    VkBlendFactor dst_alpha = VK_BLEND_FACTOR_ZERO;
    VkBlendOp alpha_op = VK_BLEND_OP_ADD;
    VkColorComponentFlags write_mask = VK_COLOR_COMPONENT_R_BIT |
                                       VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT |
                                       VK_COLOR_COMPONENT_A_BIT;
};

// Graphics pipelines created here always use dynamic viewport and scissor.
// A non-null pipeline_layout is borrowed. Otherwise one layout is created
// from set_layout and the push-constant range, and this object destroys it.
struct GraphicsDesc {
    std::span<const std::byte> vertex_spirv;
    std::span<const std::byte> fragment_spirv;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    std::uint32_t push_bytes = 0;
    VkShaderStageFlags push_stages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    std::uint32_t vertex_stride = 0;
    std::span<const VertexAttribute> attributes;
    bool depth_test = false;
    bool depth_write = false;
    VkCompareOp depth_compare = VK_COMPARE_OP_LESS;
    bool depth_bias = false;
    float depth_bias_constant = 0.F;
    float depth_bias_slope = 0.F;
    VkCullModeFlags cull = VK_CULL_MODE_NONE;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    ColorBlend blend{};
    VkRenderPass render_pass = VK_NULL_HANDLE;
};

class GraphicsPipeline {
public:
    GraphicsPipeline() = default;
    GraphicsPipeline(VkDevice device, const GraphicsDesc& desc);
    ~GraphicsPipeline();
    GraphicsPipeline(GraphicsPipeline&& other) noexcept;
    GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept;
    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;

    [[nodiscard]] bool valid() const noexcept { return handle_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkPipeline handle() const noexcept { return handle_; }
    [[nodiscard]] VkPipelineLayout layout() const noexcept { return layout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline handle_ = VK_NULL_HANDLE;
    bool owns_layout_ = false;
};

}  // namespace photara::vk
