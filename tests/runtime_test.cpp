#include "photara_vk/photara_vk.hpp"

#include "add.hlsl.embedded.hpp"
#include "ray_query.hlsl.embedded.hpp"
#include "triangle.ps.hlsl.embedded.hpp"
#include "triangle.vs.hlsl.embedded.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace photara::vk;

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(const float actual, const float expected, const float eps,
                  const char* message) {
    if (std::abs(actual - expected) > eps) {
        throw std::runtime_error(
            std::string(message) + " actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

std::span<const std::byte> add_spirv() {
    return std::as_bytes(std::span{add_hlsl_spv});
}

Device make_device() {
    DeviceRequest request;
    request.want.push_descriptors = true;
    return Device::create(request);
}

void test_enumerate_and_create() {
    const auto devices = Device::enumerate();
    require(!devices.empty(), "enumerate found no devices");
    auto device = make_device();
    require(device.valid(), "created device");
    require(device.handle() != VK_NULL_HANDLE, "logical device");
    require(device.queue() != VK_NULL_HANDLE, "queue");
    require(device.queue_family() != VK_QUEUE_FAMILY_IGNORED, "queue family");
    require(!device.caps().name.empty(), "device name");
    std::cout << "device: " << device.caps().name
              << " push_descriptors=" << device.caps().enabled.push_descriptors
              << "\n";
}

void test_host_buffer_roundtrip() {
    auto device = make_device();
    const std::vector<float> host{1.F, 2.F, 3.F, 4.F};
    auto buffer = device.create_buffer(
        host.size() * sizeof(float), MemoryKind::host_visible);
    buffer.upload(host.data(), host.size() * sizeof(float));
    std::vector<float> back(host.size());
    buffer.download(back.data(), back.size() * sizeof(float));
    require(back == host, "host buffer roundtrip");
}

void test_fill_and_copy() {
    auto device = make_device();
    auto gpu = device.create_buffer(16, MemoryKind::device_local);
    auto host = device.create_buffer(16, MemoryKind::host_cached);
    auto encoder = device.encoder();
    encoder.fill_u32(gpu, 0, 16, 0x3f800000u);  // 1.0f
    encoder.copy(host, 0, gpu, 0, 16);
    encoder.submit_wait();
    std::vector<float> values(4);
    host.download(values.data(), values.size() * sizeof(float));
    for (float value : values) require_near(value, 1.F, 1e-6F, "fill_u32");
}

void test_compute_add() {
    auto device = make_device();
    constexpr std::uint32_t kCount = 256;
    std::vector<float> input(kCount);
    for (std::uint32_t i = 0; i < kCount; ++i) input[i] = static_cast<float>(i);
    auto staging = device.create_buffer(
        kCount * sizeof(float), MemoryKind::host_visible);
    auto gpu = device.create_buffer(kCount * sizeof(float), MemoryKind::device_local);
    auto readback = device.create_buffer(
        kCount * sizeof(float), MemoryKind::host_cached);
    staging.upload(input.data(), input.size() * sizeof(float));

    auto pipeline = device.create_compute(add_spirv(), 1, 8);
    struct Push {
        std::uint32_t count;
        float addend;
    } push{kCount, 3.5F};

    auto encoder = device.encoder();
    encoder.copy(gpu, 0, staging, 0, kCount * sizeof(float));
    const BufferBinding bindings[] = {binding(gpu)};
    encoder.dispatch(
        pipeline, bindings, &push, sizeof(push), ceil_div(kCount, 64));
    encoder.copy(readback, 0, gpu, 0, kCount * sizeof(float));
    encoder.submit_wait();

    std::vector<float> output(kCount);
    readback.download(output.data(), output.size() * sizeof(float));
    for (std::uint32_t i = 0; i < kCount; ++i) {
        require_near(output[i], input[i] + 3.5F, 1e-5F, "compute add");
    }
}

void test_dispatch_indirect() {
    auto device = make_device();
    constexpr std::uint32_t kCount = 256;
    std::vector<float> input(kCount);
    for (std::uint32_t i = 0; i < kCount; ++i) input[i] = static_cast<float>(i);
    auto staging = device.create_buffer(
        kCount * sizeof(float), MemoryKind::host_visible);
    auto gpu = device.create_buffer(kCount * sizeof(float), MemoryKind::device_local);
    auto readback = device.create_buffer(
        kCount * sizeof(float), MemoryKind::host_cached);
    auto indirect = device.create_buffer(
        sizeof(VkDispatchIndirectCommand), MemoryKind::device_local,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    staging.upload(input.data(), input.size() * sizeof(float));
    auto pipeline = device.create_compute(add_spirv(), 1, 8);
    struct Push {
        std::uint32_t count;
        float addend;
    } push{kCount, 2.25F};

    auto encoder = device.encoder(BarrierPolicy::after_compute_indirect);
    encoder.copy(gpu, 0, staging, 0, kCount * sizeof(float));
    encoder.fill_u32(indirect, 0, 4, ceil_div(kCount, 64));
    encoder.fill_u32(indirect, 4, 4, 1);
    encoder.fill_u32(indirect, 8, 4, 1);
    const BufferBinding bindings[] = {binding(gpu)};
    encoder.dispatch_indirect(pipeline, bindings, &push, sizeof(push), indirect, 0);
    encoder.copy(readback, 0, gpu, 0, kCount * sizeof(float));
    encoder.submit_wait();

    std::vector<float> output(kCount);
    readback.download(output.data(), output.size() * sizeof(float));
    for (std::uint32_t i = 0; i < kCount; ++i) {
        require_near(output[i], input[i] + 2.25F, 1e-5F, "dispatch_indirect add");
    }
}

void test_encoder_ring() {
    auto device = make_device();
    constexpr std::uint32_t kCount = 64;
    auto gpu_a = device.create_buffer(kCount * sizeof(float), MemoryKind::device_local);
    auto gpu_b = device.create_buffer(kCount * sizeof(float), MemoryKind::device_local);
    auto host = device.create_buffer(
        2 * kCount * sizeof(float), MemoryKind::host_cached);
    auto pipeline = device.create_compute(add_spirv(), 1, 8);
    struct Push {
        std::uint32_t count;
        float addend;
    };

    EncoderRing ring(device, 2, BarrierPolicy::after_compute);
    {
        auto& cmd = ring.acquire();
        cmd.fill_u32(gpu_a, 0, kCount * sizeof(float), 0);
        Push push{kCount, 1.5F};
        const BufferBinding bindings[] = {binding(gpu_a)};
        cmd.dispatch(pipeline, bindings, &push, sizeof(push), ceil_div(kCount, 64));
        ring.submit();
    }
    {
        auto& cmd = ring.acquire();
        cmd.fill_u32(gpu_b, 0, kCount * sizeof(float), 0);
        Push push{kCount, 4.0F};
        const BufferBinding bindings[] = {binding(gpu_b)};
        cmd.dispatch(pipeline, bindings, &push, sizeof(push), ceil_div(kCount, 64));
        ring.submit_wait();
    }
    {
        auto& cmd = ring.acquire();
        cmd.copy(host, 0, gpu_a, 0, kCount * sizeof(float));
        cmd.copy(host, kCount * sizeof(float), gpu_b, 0, kCount * sizeof(float));
        ring.submit_wait();
    }

    std::vector<float> output(2 * kCount);
    host.download(output.data(), output.size() * sizeof(float));
    for (std::uint32_t i = 0; i < kCount; ++i) {
        require_near(output[i], 1.5F, 1e-5F, "encoder ring slot 0");
        require_near(output[kCount + i], 4.0F, 1e-5F, "encoder ring slot 1");
    }
}

void test_adopt() {
    auto owner = make_device();
    DeviceRequest request;
    request.want.push_descriptors = owner.caps().enabled.push_descriptors;
    auto adopted = Device::adopt(owner.handles());
    require(adopted.handle() == owner.handle(), "adopted same device");
    require(!adopted.owns_device(), "adopted does not own the device");

    constexpr std::uint32_t kCount = 64;
    auto staging = adopted.create_buffer(
        kCount * sizeof(float), MemoryKind::host_visible);
    auto gpu = adopted.create_buffer(
        kCount * sizeof(float), MemoryKind::device_local);
    std::vector<float> input(kCount, 2.F);
    staging.upload(input.data(), input.size() * sizeof(float));
    auto pipeline = adopted.create_compute(add_spirv(), 1, 8);
    struct Push {
        std::uint32_t count;
        float addend;
    } push{kCount, 1.F};
    auto encoder = adopted.encoder();
    encoder.copy(gpu, 0, staging, 0, kCount * sizeof(float));
    const BufferBinding bindings[] = {binding(gpu)};
    encoder.dispatch(
        pipeline, bindings, &push, sizeof(push), ceil_div(kCount, 64));
    encoder.copy(staging, 0, gpu, 0, kCount * sizeof(float));
    encoder.submit_wait();
    std::vector<float> output(kCount);
    staging.download(output.data(), output.size() * sizeof(float));
    for (float value : output) require_near(value, 3.F, 1e-5F, "adopted add");
}

std::span<const std::byte> ray_spirv() {
    return std::as_bytes(std::span{ray_query_hlsl_spv});
}

std::uint32_t trace_ray(
    Device& device, const TriangleScene& scene, const ComputePipeline& pipeline,
    Buffer& hits, const float origin_x, const float origin_y, const float origin_z,
    const float direction_x, const float direction_y, const float direction_z) {
    const std::uint32_t zero = 0;
    hits.upload(&zero, sizeof(zero));
    struct Push {
        float origin_tmin[4];
        float direction_tmax[4];
    } push{{origin_x, origin_y, origin_z, 0.001F},
           {direction_x, direction_y, direction_z, 10.F}};
    auto encoder = device.encoder(BarrierPolicy::none);
    const VkCommandBuffer command = encoder.native();
    encoder.barrier(
        VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_HOST_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    VkAccelerationStructureKHR top = scene.top_level();
    VkWriteDescriptorSetAccelerationStructureKHR acceleration{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    acceleration.accelerationStructureCount = 1;
    acceleration.pAccelerationStructures = &top;
    const VkDescriptorBufferInfo buffer_info{hits.handle(), 0, hits.size()};
    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &acceleration;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &buffer_info;
    const auto push_descriptors = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        vkGetDeviceProcAddr(device.handle(), "vkCmdPushDescriptorSetKHR"));
    require(push_descriptors != nullptr, "push descriptor entry point");
    push_descriptors(
        command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 2, writes);
    vkCmdPushConstants(
        command, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(command, 1, 1, 1);
    encoder.barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    encoder.submit_wait();
    std::uint32_t hit = 0;
    hits.download(&hit, sizeof(hit));
    return hit;
}

void test_triangle_scene() {
    DeviceRequest request;
    request.want.ray_query = true;
    auto device = Device::create(request);
    if (!device.caps().enabled.ray_query) {
        std::cout << "triangle scene: skipped (no ray query)\n";
        return;
    }
    require(device.caps().enabled.buffer_device_address, "ray query enables device address");
    require(device.caps().enabled.push_descriptors, "ray query test needs push descriptors");
    bool rejected = false;
    try {
        const float none[] = {0.F};
        (void)device.create_triangle_scene(none, {});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "empty ray-query geometry is rejected");

    const float positions[] = {
        0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 1.F, 0.F,
    };
    const std::uint32_t indices[] = {0, 1, 2};
    auto scene = device.create_triangle_scene(positions, indices);
    require(scene.bottom_level() != VK_NULL_HANDLE, "bottom level");
    require(scene.top_level() != VK_NULL_HANDLE, "top level");
    TriangleScene moved = std::move(scene);
    require(!scene.valid(), "moved-from scene");
    require(moved.valid(), "moved scene");

    const VkDescriptorType types[] = {
        VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    auto pipeline = device.create_compute(ray_spirv(), types, 32);
    auto hits = device.create_buffer(sizeof(std::uint32_t), MemoryKind::host_cached);
    require(
        trace_ray(device, moved, pipeline, hits, 0.2F, 0.2F, 1.F, 0.F, 0.F, -1.F) == 1,
        "ray hits the triangle");
    require(
        trace_ray(device, moved, pipeline, hits, 2.F, 2.F, 1.F, 0.F, 0.F, -1.F) == 0,
        "ray misses the triangle");
    std::cout << "triangle scene: hit and miss\n";
}

void test_graphics_triangle() {
    auto device = make_device();
    if (!device.caps().enabled.graphics) {
        std::cout << "graphics triangle: skipped\n";
        return;
    }
    AttachmentDesc color;
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.store_op = VK_ATTACHMENT_STORE_OP_STORE;
    color.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.final_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    SubpassDependencyDesc incoming;
    incoming.src_subpass = VK_SUBPASS_EXTERNAL;
    incoming.dst_subpass = 0;
    incoming.src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    incoming.dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    incoming.dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    SubpassDependencyDesc outgoing;
    outgoing.src_subpass = 0;
    outgoing.dst_subpass = VK_SUBPASS_EXTERNAL;
    outgoing.src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    outgoing.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    outgoing.src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    outgoing.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
    const AttachmentDesc attachments[] = {color};
    const SubpassDependencyDesc dependencies[] = {incoming, outgoing};
    auto pass = device.create_render_pass(attachments, dependencies);
    ImageDesc image_desc;
    image_desc.width = 8;
    image_desc.height = 8;
    image_desc.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_desc.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    auto image = device.create_image(image_desc);
    auto view = device.create_image_view(
        image.handle(), image.format(), VK_IMAGE_ASPECT_COLOR_BIT);
    const VkImageView views[] = {view.handle()};
    auto framebuffer = device.create_framebuffer(pass, views, 8, 8);
    GraphicsDesc desc;
    desc.vertex_spirv = std::as_bytes(std::span{triangle_vs_hlsl_spv});
    desc.fragment_spirv = std::as_bytes(std::span{triangle_ps_hlsl_spv});
    desc.render_pass = pass.handle();
    auto pipeline = device.create_graphics(desc);
    require(pipeline.valid(), "graphics pipeline");
    auto readback = device.create_buffer(8U * 8U * 4U, MemoryKind::host_cached);

    auto encoder = device.encoder(BarrierPolicy::none);
    const VkCommandBuffer command = encoder.native();
    VkClearValue clear{};
    clear.color.float32[3] = 1.F;
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = pass.handle();
    begin.framebuffer = framebuffer.handle();
    begin.renderArea.extent = {8, 8};
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    vkCmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{};
    viewport.width = 8.F;
    viewport.height = 8.F;
    viewport.maxDepth = 1.F;
    const VkRect2D scissor{{0, 0}, {8, 8}};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdEndRenderPass(command);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {8, 8, 1};
    vkCmdCopyImageToBuffer(
        command, image.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle(),
        1, &region);
    encoder.barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    encoder.submit_wait();

    std::vector<std::uint8_t> pixels(8U * 8U * 4U);
    readback.download(pixels.data(), pixels.size());
    require(pixels[0] == 255 && pixels[1] == 0 && pixels[2] == 0 && pixels[3] == 255,
            "graphics triangle is red");
    std::cout << "graphics triangle: red\n";
}

}  // namespace

int main() {
    try {
        test_enumerate_and_create();
        test_host_buffer_roundtrip();
        test_fill_and_copy();
        test_compute_add();
        test_dispatch_indirect();
        test_encoder_ring();
        test_adopt();
        test_triangle_scene();
        test_graphics_triangle();
    } catch (const std::exception& error) {
        std::cerr << "photara_vk runtime test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "photara_vk runtime test: ok\n";
    return 0;
}
