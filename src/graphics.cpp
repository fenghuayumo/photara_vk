#include "photara_vk/graphics.hpp"

#include "device_impl.hpp"
#include "photara_vk/check.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace photara::vk {
namespace {

std::uint32_t find_memory_type(
    const VkPhysicalDevice physical, const std::uint32_t type_bits,
    const VkMemoryPropertyFlags required) {
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

VkShaderModule make_module(const VkDevice device, const std::span<const std::byte> spirv) {
    if (spirv.empty() || spirv.size() % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error("Invalid embedded SPIR-V bytecode");
    }
    std::vector<std::uint32_t> words(spirv.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), spirv.data(), spirv.size());
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = words.size() * sizeof(std::uint32_t);
    info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    check_vk(vkCreateShaderModule(device, &info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

struct ShaderModule {
    VkDevice device = VK_NULL_HANDLE;
    VkShaderModule handle = VK_NULL_HANDLE;
    ~ShaderModule() {
        if (device != VK_NULL_HANDLE && handle != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, handle, nullptr);
        }
    }
    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule() = default;
    ShaderModule(ShaderModule&& other) noexcept
        : device(other.device), handle(other.handle) {
        other.device = VK_NULL_HANDLE;
        other.handle = VK_NULL_HANDLE;
    }
    ShaderModule& operator=(ShaderModule&&) = delete;
};

}  // namespace

Image::Image(
    const VkPhysicalDevice physical, const VkDevice device, const ImageDesc& desc)
    : device_(device),
      width_(desc.width),
      height_(desc.height),
      format_(desc.format) {
    if (device_ == VK_NULL_HANDLE || desc.width == 0 || desc.height == 0 ||
        desc.format == VK_FORMAT_UNDEFINED || desc.usage == 0) {
        throw std::invalid_argument("Image description is incomplete");
    }
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = desc.format;
    info.extent = {desc.width, desc.height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = desc.usage;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    check_vk(vkCreateImage(device_, &info, nullptr, &handle_), "vkCreateImage");
    try {
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, handle_, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex =
            find_memory_type(physical, requirements.memoryTypeBits, desc.memory);
        check_vk(vkAllocateMemory(device_, &allocation, nullptr, &memory_),
                 "vkAllocateMemory");
        try {
            check_vk(vkBindImageMemory(device_, handle_, memory_, 0), "vkBindImageMemory");
        } catch (...) {
            vkFreeMemory(device_, memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
            throw;
        }
    } catch (...) {
        vkDestroyImage(device_, handle_, nullptr);
        handle_ = VK_NULL_HANDLE;
        throw;
    }
}

Image::~Image() {
    if (device_ == VK_NULL_HANDLE) return;
    if (handle_ != VK_NULL_HANDLE) vkDestroyImage(device_, handle_, nullptr);
    if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
}

Image::Image(Image&& other) noexcept { *this = std::move(other); }

Image& Image::operator=(Image&& other) noexcept {
    if (this == &other) return *this;
    std::swap(device_, other.device_);
    std::swap(handle_, other.handle_);
    std::swap(memory_, other.memory_);
    std::swap(width_, other.width_);
    std::swap(height_, other.height_);
    std::swap(format_, other.format_);
    return *this;
}

ImageView::ImageView(
    const VkDevice device, const VkImage image, const VkFormat format,
    const VkImageAspectFlags aspect)
    : device_(device) {
    if (device == VK_NULL_HANDLE || image == VK_NULL_HANDLE ||
        format == VK_FORMAT_UNDEFINED || aspect == 0) {
        throw std::invalid_argument("Image view description is incomplete");
    }
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange = {aspect, 0, 1, 0, 1};
    check_vk(vkCreateImageView(device_, &info, nullptr, &handle_), "vkCreateImageView");
}

ImageView::~ImageView() {
    if (device_ != VK_NULL_HANDLE && handle_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, handle_, nullptr);
    }
}

ImageView::ImageView(ImageView&& other) noexcept { *this = std::move(other); }

ImageView& ImageView::operator=(ImageView&& other) noexcept {
    if (this == &other) return *this;
    std::swap(device_, other.device_);
    std::swap(handle_, other.handle_);
    return *this;
}

RenderPass::RenderPass(
    const VkDevice device, const std::span<const AttachmentDesc> attachments,
    const std::span<const SubpassDependencyDesc> dependencies)
    : device_(device) {
    if (device_ == VK_NULL_HANDLE || attachments.empty()) {
        throw std::invalid_argument("Render pass needs a device and an attachment");
    }
    std::vector<VkAttachmentDescription> descriptions(attachments.size());
    std::vector<VkAttachmentReference> colors;
    VkAttachmentReference depth_ref{};
    bool has_depth = false;
    for (std::uint32_t index = 0; index < attachments.size(); ++index) {
        const AttachmentDesc& attachment = attachments[index];
        if (attachment.format == VK_FORMAT_UNDEFINED) {
            throw std::invalid_argument("Render pass attachment format is undefined");
        }
        descriptions[index].format = attachment.format;
        descriptions[index].samples = VK_SAMPLE_COUNT_1_BIT;
        descriptions[index].loadOp = attachment.load_op;
        descriptions[index].storeOp = attachment.store_op;
        descriptions[index].stencilLoadOp = attachment.stencil_load_op;
        descriptions[index].stencilStoreOp = attachment.stencil_store_op;
        descriptions[index].initialLayout = attachment.initial_layout;
        descriptions[index].finalLayout = attachment.final_layout;
        if (attachment.depth) {
            if (has_depth) {
                throw std::invalid_argument("Render pass has more than one depth attachment");
            }
            has_depth = true;
            depth_ref.attachment = index;
            depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        } else {
            colors.push_back(
                VkAttachmentReference{index, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        }
    }

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<std::uint32_t>(colors.size());
    subpass.pColorAttachments = colors.empty() ? nullptr : colors.data();
    subpass.pDepthStencilAttachment = has_depth ? &depth_ref : nullptr;

    std::vector<VkSubpassDependency> deps(dependencies.size());
    for (std::size_t index = 0; index < dependencies.size(); ++index) {
        const SubpassDependencyDesc& dependency = dependencies[index];
        deps[index].srcSubpass = dependency.src_subpass;
        deps[index].dstSubpass = dependency.dst_subpass;
        deps[index].srcStageMask = dependency.src_stage;
        deps[index].dstStageMask = dependency.dst_stage;
        deps[index].srcAccessMask = dependency.src_access;
        deps[index].dstAccessMask = dependency.dst_access;
    }

    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = static_cast<std::uint32_t>(descriptions.size());
    info.pAttachments = descriptions.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = static_cast<std::uint32_t>(deps.size());
    info.pDependencies = deps.empty() ? nullptr : deps.data();
    check_vk(vkCreateRenderPass(device_, &info, nullptr, &handle_), "vkCreateRenderPass");
}

RenderPass::~RenderPass() {
    if (device_ != VK_NULL_HANDLE && handle_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device_, handle_, nullptr);
    }
}

RenderPass::RenderPass(RenderPass&& other) noexcept { *this = std::move(other); }

RenderPass& RenderPass::operator=(RenderPass&& other) noexcept {
    if (this == &other) return *this;
    std::swap(device_, other.device_);
    std::swap(handle_, other.handle_);
    return *this;
}

Framebuffer::Framebuffer(
    const VkDevice device, const VkRenderPass render_pass,
    const std::span<const VkImageView> attachments, const std::uint32_t width,
    const std::uint32_t height)
    : device_(device) {
    if (device == VK_NULL_HANDLE || render_pass == VK_NULL_HANDLE ||
        attachments.empty() || width == 0 || height == 0) {
        throw std::invalid_argument("Framebuffer description is incomplete");
    }
    VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    info.renderPass = render_pass;
    info.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    info.pAttachments = attachments.data();
    info.width = width;
    info.height = height;
    info.layers = 1;
    check_vk(vkCreateFramebuffer(device_, &info, nullptr, &handle_), "vkCreateFramebuffer");
}

Framebuffer::~Framebuffer() {
    if (device_ != VK_NULL_HANDLE && handle_ != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device_, handle_, nullptr);
    }
}

Framebuffer::Framebuffer(Framebuffer&& other) noexcept { *this = std::move(other); }

Framebuffer& Framebuffer::operator=(Framebuffer&& other) noexcept {
    if (this == &other) return *this;
    std::swap(device_, other.device_);
    std::swap(handle_, other.handle_);
    return *this;
}

GraphicsPipeline::GraphicsPipeline(const VkDevice device, const GraphicsDesc& desc)
    : device_(device), layout_(desc.pipeline_layout), owns_layout_(false) {
    if (device_ == VK_NULL_HANDLE || desc.render_pass == VK_NULL_HANDLE) {
        throw std::invalid_argument("Graphics pipeline needs a device and a render pass");
    }
    if (desc.vertex_spirv.empty() || desc.fragment_spirv.empty()) {
        throw std::invalid_argument("Graphics pipeline needs vertex and fragment SPIR-V");
    }
    if ((desc.vertex_stride == 0) != desc.attributes.empty()) {
        throw std::invalid_argument(
            "Graphics vertex stride and attributes must be set together");
    }
    if (layout_ == VK_NULL_HANDLE) {
        if (desc.push_bytes > 0 && desc.push_stages == 0) {
            throw std::invalid_argument("Graphics push constants need a stage");
        }
        VkPushConstantRange push{};
        push.stageFlags = desc.push_stages;
        push.size = desc.push_bytes;
        VkPipelineLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = desc.set_layout == VK_NULL_HANDLE ? 0U : 1U;
        layout_info.pSetLayouts =
            desc.set_layout == VK_NULL_HANDLE ? nullptr : &desc.set_layout;
        layout_info.pushConstantRangeCount = desc.push_bytes == 0 ? 0U : 1U;
        layout_info.pPushConstantRanges = desc.push_bytes == 0 ? nullptr : &push;
        check_vk(
            vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_),
            "vkCreatePipelineLayout");
        owns_layout_ = true;
    }

    try {
    ShaderModule vertex;
    ShaderModule fragment;
    vertex.device = device_;
    fragment.device = device_;
    vertex.handle = make_module(device_, desc.vertex_spirv);
    fragment.handle = make_module(device_, desc.fragment_spirv);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex.handle;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment.handle;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = desc.vertex_stride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::vector<VkVertexInputAttributeDescription> attributes(desc.attributes.size());
    for (std::size_t index = 0; index < desc.attributes.size(); ++index) {
        attributes[index].location = desc.attributes[index].location;
        attributes[index].binding = 0;
        attributes[index].format = desc.attributes[index].format;
        attributes[index].offset = desc.attributes[index].offset;
    }
    VkPipelineVertexInputStateCreateInfo vertex_state{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    if (desc.vertex_stride != 0) {
        vertex_state.vertexBindingDescriptionCount = 1;
        vertex_state.pVertexBindingDescriptions = &binding;
        vertex_state.vertexAttributeDescriptionCount =
            static_cast<std::uint32_t>(attributes.size());
        vertex_state.pVertexAttributeDescriptions = attributes.data();
    }
    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = desc.topology;
    VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = desc.cull;
    raster.frontFace = desc.front_face;
    raster.lineWidth = 1.F;
    raster.depthBiasEnable = desc.depth_bias ? VK_TRUE : VK_FALSE;
    raster.depthBiasConstantFactor = desc.depth_bias ? desc.depth_bias_constant : 0.F;
    raster.depthBiasSlopeFactor = desc.depth_bias ? desc.depth_bias_slope : 0.F;
    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = desc.depth_test ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp = desc.depth_compare;
    const bool use_depth = desc.depth_test || desc.depth_write;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.blendEnable = desc.blend.enable ? VK_TRUE : VK_FALSE;
    blend_attachment.srcColorBlendFactor = desc.blend.src_color;
    blend_attachment.dstColorBlendFactor = desc.blend.dst_color;
    blend_attachment.colorBlendOp = desc.blend.color_op;
    blend_attachment.srcAlphaBlendFactor = desc.blend.src_alpha;
    blend_attachment.dstAlphaBlendFactor = desc.blend.dst_alpha;
    blend_attachment.alphaBlendOp = desc.blend.alpha_op;
    blend_attachment.colorWriteMask = desc.blend.write_mask;
    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_state;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = use_depth ? &depth : nullptr;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout_;
    info.renderPass = desc.render_pass;
    const VkResult result = vkCreateGraphicsPipelines(
        device_, VK_NULL_HANDLE, 1, &info, nullptr, &handle_);
    if (result != VK_SUCCESS) {
        if (owns_layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
        owns_layout_ = false;
        check_vk(result, "vkCreateGraphicsPipelines");
    }
    } catch (...) {
        if (owns_layout_ && layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, layout_, nullptr);
        }
        layout_ = VK_NULL_HANDLE;
        owns_layout_ = false;
        handle_ = VK_NULL_HANDLE;
        throw;
    }
}

GraphicsPipeline::~GraphicsPipeline() {
    if (device_ == VK_NULL_HANDLE) return;
    if (handle_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, handle_, nullptr);
    if (owns_layout_ && layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
    }
}

GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&& other) noexcept {
    *this = std::move(other);
}

GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&& other) noexcept {
    if (this == &other) return *this;
    std::swap(device_, other.device_);
    std::swap(layout_, other.layout_);
    std::swap(handle_, other.handle_);
    std::swap(owns_layout_, other.owns_layout_);
    return *this;
}

Image Device::create_image(const ImageDesc& desc) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return Image(impl_->physical, impl_->device, desc);
}

ImageView Device::create_image_view(
    const VkImage image, const VkFormat format, const VkImageAspectFlags aspect) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return ImageView(impl_->device, image, format, aspect);
}

RenderPass Device::create_render_pass(
    const std::span<const AttachmentDesc> attachments,
    const std::span<const SubpassDependencyDesc> dependencies) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return RenderPass(impl_->device, attachments, dependencies);
}

Framebuffer Device::create_framebuffer(
    const RenderPass& render_pass, const std::span<const VkImageView> attachments,
    const std::uint32_t width, const std::uint32_t height) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return Framebuffer(impl_->device, render_pass.handle(), attachments, width, height);
}

GraphicsPipeline Device::create_graphics(const GraphicsDesc& desc) const {
    if (!impl_) throw std::logic_error("photara_vk Device is empty");
    return GraphicsPipeline(impl_->device, desc);
}

}  // namespace photara::vk
