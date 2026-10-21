#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "add.hlsl.embedded.hpp"
#include "photara_vk/photara_vk.hpp"
#include "ray_query.hlsl.embedded.hpp"
#include "sample_image.hlsl.embedded.hpp"
#include "triangle.ps.hlsl.embedded.hpp"
#include "triangle.vs.hlsl.embedded.hpp"

namespace {

using namespace photara::vk;

void require(const bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(const float actual, const float expected, const float eps,
                  const char *message) {
    if (std::abs(actual - expected) > eps) {
        throw std::runtime_error(std::string(message) +
                                 " actual=" + std::to_string(actual) +
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
              << " push_descriptors=" << device.caps().enabled.push_descriptors << "\n";
}

void test_host_buffer_roundtrip() {
    auto device = make_device();
    const std::vector<float> host{1.F, 2.F, 3.F, 4.F};
    auto buffer =
        device.create_buffer(host.size() * sizeof(float), MemoryKind::host_visible);
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

void test_timestamp_pool() {
    auto device = make_device();
    if (device.caps().timestamp_valid_bits == 0) {
        std::cout << "timestamp pool: skipped (queue has no timestamp bits)\n";
        return;
    }
    require(device.caps().timestamp_period_nanoseconds > 0.0, "timestamp period");

    auto pool = device.create_timestamp_pool(2);
    require(pool.valid() && pool.count() == 2, "timestamp pool");
    auto destination = device.create_buffer(16, MemoryKind::device_local);
    auto encoder = device.encoder(BarrierPolicy::none);
    encoder.reset_timestamps(pool);
    encoder.write_timestamp(pool, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    encoder.fill_u32(destination, 0, 16, 0x3f800000u);
    encoder.write_timestamp(pool, 1, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    encoder.submit_wait();

    const auto timestamps = pool.results();
    require(timestamps.size() == 2, "timestamp result count");
    require(timestamps[1] >= timestamps[0], "timestamp ordering");
    require(pool.elapsed_nanoseconds(0, 1) >= 0.0, "timestamp interval");
    require(pool.elapsed_seconds(0, 1) >= 0.0, "timestamp interval seconds");
    std::cout << "timestamp pool: " << pool.elapsed_nanoseconds(0, 1) << " ns\n";
}

void test_compute_add() {
    auto device = make_device();
    constexpr std::uint32_t kCount = 256;
    std::vector<float> input(kCount);
    for (std::uint32_t i = 0; i < kCount; ++i) input[i] = static_cast<float>(i);
    auto staging =
        device.create_buffer(kCount * sizeof(float), MemoryKind::host_visible);
    auto gpu = device.create_buffer(kCount * sizeof(float), MemoryKind::device_local);
    auto readback =
        device.create_buffer(kCount * sizeof(float), MemoryKind::host_cached);
    staging.upload(input.data(), input.size() * sizeof(float));

    auto pipeline = device.create_compute(add_spirv(), 1, 8);
    struct Push {
        std::uint32_t count;
        float addend;
    } push{kCount, 3.5F};

    auto encoder = device.encoder();
    encoder.copy(gpu, 0, staging, 0, kCount * sizeof(float));
    const BufferBinding bindings[] = {binding(gpu)};
    encoder.dispatch(pipeline, bindings, &push, sizeof(push), ceil_div(kCount, 64));
    encoder.copy(readback, 0, gpu, 0, kCount * sizeof(float));
    encoder.submit_wait();

    std::vector<float> output(kCount);
    readback.download(output.data(), output.size() * sizeof(float));
    for (std::uint32_t i = 0; i < kCount; ++i) {
        require_near(output[i], input[i] + 3.5F, 1e-5F, "compute add");
    }
}

void test_sampled_image(bool push) {
    DeviceRequest request;
    request.want.push_descriptors = push;
    request.validation = true;
    request.synchronization_validation = true;
    auto device = Device::create(request);
    ImageDesc desc;
    desc.width = 2;
    desc.height = 1;
    desc.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    desc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    require(device.supports_linear_sampling(desc.format),
            "RGBA32F linear sampling unsupported");
    auto image = device.create_image(desc);
    auto view = device.create_image_view(image.handle(), desc.format,
                                         VK_IMAGE_ASPECT_COLOR_BIT);
    auto sampler = device.create_sampler();
    auto staging = device.create_buffer(32, MemoryKind::host_visible);
    const float texels[] = {2, 0, 0, 1, 0, 4, 0, 1};
    staging.upload(texels, sizeof(texels));
    auto output = device.create_buffer(16, MemoryKind::device_local);
    auto readback = device.create_buffer(16, MemoryKind::host_cached);
    const VkDescriptorType types[] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                      VK_DESCRIPTOR_TYPE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    auto pipeline = device.create_compute(
        std::as_bytes(std::span{sample_image_hlsl_spv}), types, 0);
    auto encoder = device.encoder(BarrierPolicy::none);
    encoder.transition_image(
        image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        VK_ACCESS_TRANSFER_WRITE_BIT);
    encoder.copy_to_image(image, staging);
    encoder.transition_image(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    const DescriptorBinding bindings[] = {binding(view), binding(sampler),
                                          binding(output)};
    encoder.dispatch(pipeline, bindings, nullptr, 0, 1);
    encoder.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT);
    encoder.copy(readback, 0, output, 0, 16);
    encoder.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    encoder.submit_wait();
    float values[4]{};
    readback.download(values, sizeof(values));
    require_near(values[0], 1, 1e-6f, "Image red interpolation");
    require_near(values[1], 2, 1e-6f, "Image green interpolation");
    require_near(values[3], 1, 1e-6f, "Image alpha interpolation");
    require(device.validation_error_count() == 0,
            "Image upload/sampling validation errors");
}

template <class F>
void require_invalid(F action, const char *message) {
    try {
        action();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error(message);
}

void test_pipeline_cache(bool push_descriptors) {
    DeviceRequest request;
    request.validation = true;
    request.synchronization_validation = true;
    request.want.push_descriptors = push_descriptors;
    auto device = Device::create(request);
    auto cache = device.create_pipeline_cache();
    auto first = device.create_compute(add_spirv(), 1, 8, &cache);
    auto bytes = cache.data();
    require(bytes.size() >= sizeof(VkPipelineCacheHeaderVersionOne),
            "cache has header");
    auto restored = device.create_pipeline_cache(bytes);
    auto pipeline = device.create_compute(add_spirv(), 1, 8, &restored);
    auto moved = std::move(restored);
    require(!restored.valid() && moved.valid(), "cache move construction");
    cache = std::move(moved);
    require(!moved.valid() && cache.valid(), "cache move assignment");

    // A cache may be shared by builders and serialized at the same time.
    std::exception_ptr worker_error;
    std::jthread builder([&] {
        try {
            auto built = device.create_compute(add_spirv(), 1, 8, &cache);
        } catch (...) {
            worker_error = std::current_exception();
        }
    });
    const auto concurrent_data = cache.data();
    builder.join();
    if (worker_error) std::rethrow_exception(worker_error);
    require(!concurrent_data.empty(), "concurrent cache serialization");

    auto foreign = Device::create(request);
    require_invalid([&] { (void)foreign.create_compute(add_spirv(), 1, 8, &cache); },
                    "foreign cache accepted");
    PipelineCache empty;
    require_invalid([&] { (void)device.create_compute(add_spirv(), 1, 8, &empty); },
                    "empty cache accepted");
    require_invalid(
        [&] { (void)device.create_pipeline_cache(std::span(bytes).first(4)); },
        "truncated cache accepted");
    const std::size_t offsets[] = {
        offsetof(VkPipelineCacheHeaderVersionOne, headerSize),
        offsetof(VkPipelineCacheHeaderVersionOne, headerVersion),
        offsetof(VkPipelineCacheHeaderVersionOne, vendorID),
        offsetof(VkPipelineCacheHeaderVersionOne, deviceID),
        offsetof(VkPipelineCacheHeaderVersionOne, pipelineCacheUUID)};
    for (auto offset : offsets) {
        auto invalid = bytes;
        invalid[offset] ^= std::byte{0xff};
        require_invalid([&] { (void)device.create_pipeline_cache(invalid); },
                        "incompatible cache header accepted");
    }
    require_invalid([&] { (void)device.create_compute({}, 1, 8, &cache); },
                    "invalid SPIR-V accepted");
    require_invalid([&] { (void)device.create_compute(add_spirv(), 1, 3, &cache); },
                    "unaligned push constants accepted");

    // Pipelines do not depend on the cache after creation.
    cache = PipelineCache{};
    auto gpu = device.create_buffer(64 * sizeof(float), MemoryKind::device_local);
    auto output = device.create_buffer(64 * sizeof(float), MemoryKind::host_cached);
    auto encoder = device.encoder();
    encoder.fill_u32(gpu, 0, gpu.size(), 0);
    struct Push {
        std::uint32_t count;
        float addend;
    } push{64, 6.25F};
    const DescriptorBinding bindings[] = {binding(gpu)};
    encoder.dispatch(pipeline, bindings, &push, sizeof(push), 1);
    encoder.copy(output, 0, gpu, 0, gpu.size());
    encoder.submit_wait();
    std::vector<float> values(64);
    output.download(values.data(), values.size() * sizeof(float));
    for (float value : values) require_near(value, 6.25F, 1e-5F, "cached compute");
    require(device.validation_error_count() == 0, "pipeline cache validation errors");
    require(foreign.validation_error_count() == 0, "foreign cache validation errors");
    std::cout << "pipeline cache: restored/dispatch/validation ok, push="
              << device.caps().enabled.push_descriptors << "\n";
}

void test_resource_device_lifetime() {
    // Each returned resource is the last owning reference to its original device.
    auto buffer = [] {
        auto device = make_device();
        return device.create_buffer(16, MemoryKind::host_visible);
    }();
    const std::uint32_t value = 42;
    std::uint32_t result = 0;
    buffer.upload(&value, sizeof(value));
    buffer.download(&result, sizeof(result));
    require(result == value, "buffer retained device");
    auto cache = [] {
        auto device = make_device();
        return device.create_pipeline_cache();
    }();
    require(!cache.data().empty(), "cache retained device");
    struct OrphanPipeline {
        ComputePipeline pipeline;
        ExternalDevice handles;
    };
    auto orphan = [] {
        auto device = make_device();
        return OrphanPipeline{device.create_compute(add_spirv(), 1, 8),
                              device.handles()};
    }();
    {
        // Adopted wrappers borrow the native device. Only orphan.pipeline owns it.
        auto borrowed = Device::adopt(orphan.handles);
        auto output =
            borrowed.create_buffer(64 * sizeof(float), MemoryKind::host_visible);
        auto encoder = borrowed.encoder();
        encoder.fill_u32(output, 0, output.size(), 0);
        struct Push {
            std::uint32_t count;
            float addend;
        } push{64, 2.5F};
        const DescriptorBinding bindings[] = {binding(output)};
        encoder.dispatch(orphan.pipeline, bindings, &push, sizeof(push), 1);
        encoder.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_ACCESS_HOST_READ_BIT);
        encoder.submit_wait();
        std::vector<float> values(64);
        output.download(values.data(), values.size() * sizeof(float));
        for (float v : values) require_near(v, 2.5F, 1e-5F, "pipeline retained device");
    }
    auto pool = [] {
        auto device = make_device();
        if (device.caps().timestamp_valid_bits == 0) return TimestampPool{};
        auto queries = device.create_timestamp_pool(2);
        auto encoder = device.encoder();
        encoder.reset_timestamps(queries);
        encoder.write_timestamp(queries, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        encoder.write_timestamp(queries, 1, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        encoder.submit_wait();
        return queries;
    }();
    if (pool.valid()) {
        TimestampPool moved;
        moved = std::move(pool);
        require(!pool.valid() && moved.results().size() == 2,
                "timestamp retained device on move");
    }
    auto image = [] {
        auto device = make_device();
        return device.create_image(ImageDesc{});
    }();
    require(image.valid(), "image retained device");
}

void test_dispatch_indirect() {
    auto device = make_device();
    constexpr std::uint32_t kCount = 256;
    std::vector<float> input(kCount);
    for (std::uint32_t i = 0; i < kCount; ++i) input[i] = static_cast<float>(i);
    auto staging =
        device.create_buffer(kCount * sizeof(float), MemoryKind::host_visible);
    auto gpu = device.create_buffer(kCount * sizeof(float), MemoryKind::device_local);
    auto readback =
        device.create_buffer(kCount * sizeof(float), MemoryKind::host_cached);
    auto indirect = device.create_buffer(sizeof(VkDispatchIndirectCommand),
                                         MemoryKind::device_local,
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
    auto host =
        device.create_buffer(2 * kCount * sizeof(float), MemoryKind::host_cached);
    auto pipeline = device.create_compute(add_spirv(), 1, 8);
    struct Push {
        std::uint32_t count;
        float addend;
    };

    EncoderRing ring(device, 2, BarrierPolicy::after_compute);
    {
        auto &cmd = ring.acquire();
        cmd.fill_u32(gpu_a, 0, kCount * sizeof(float), 0);
        Push push{kCount, 1.5F};
        const BufferBinding bindings[] = {binding(gpu_a)};
        cmd.dispatch(pipeline, bindings, &push, sizeof(push), ceil_div(kCount, 64));
        ring.submit();
    }
    {
        auto &cmd = ring.acquire();
        cmd.fill_u32(gpu_b, 0, kCount * sizeof(float), 0);
        Push push{kCount, 4.0F};
        const BufferBinding bindings[] = {binding(gpu_b)};
        cmd.dispatch(pipeline, bindings, &push, sizeof(push), ceil_div(kCount, 64));
        ring.submit_wait();
    }
    {
        auto &cmd = ring.acquire();
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
    require(&adopted.queue_mutex() == &owner.queue_mutex(),
            "adopted queue shares the owner mutex");

    constexpr std::uint32_t kCount = 64;
    auto staging =
        adopted.create_buffer(kCount * sizeof(float), MemoryKind::host_visible);
    auto gpu = adopted.create_buffer(kCount * sizeof(float), MemoryKind::device_local);
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
    encoder.dispatch(pipeline, bindings, &push, sizeof(push), ceil_div(kCount, 64));
    encoder.copy(staging, 0, gpu, 0, kCount * sizeof(float));
    encoder.submit_wait();
    std::vector<float> output(kCount);
    staging.download(output.data(), output.size() * sizeof(float));
    for (float value : output) require_near(value, 3.F, 1e-5F, "adopted add");
}

std::span<const std::byte> ray_spirv() {
    return std::as_bytes(std::span{ray_query_hlsl_spv});
}

std::uint32_t trace_ray(Device &device, const TriangleScene &scene,
                        const ComputePipeline &pipeline, Buffer &hits,
                        const float origin_x, const float origin_y,
                        const float origin_z, const float direction_x,
                        const float direction_y, const float direction_z) {
    const std::uint32_t zero = 0;
    hits.upload(&zero, sizeof(zero));
    struct Push {
        float origin_tmin[4];
        float direction_tmax[4];
    } push{{origin_x, origin_y, origin_z, 0.001F},
           {direction_x, direction_y, direction_z, 10.F}};
    auto encoder = device.encoder(BarrierPolicy::none);
    encoder.barrier(VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_HOST_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    const DescriptorBinding bindings[] = {binding(scene), binding(hits)};
    encoder.dispatch(pipeline, bindings, &push, sizeof(push), 1);
    encoder.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    encoder.submit_wait();
    std::uint32_t hit = 0;
    hits.download(&hit, sizeof(hit));
    return hit;
}

void test_triangle_scene(const bool push_descriptors) {
    DeviceRequest request;
    request.validation = true;
    request.synchronization_validation = true;
    request.want.push_descriptors = push_descriptors;
    request.want.ray_query = true;
    auto device = Device::create(request);
    if (!device.caps().enabled.ray_query) {
        std::cout << "triangle scene: skipped (no ray query)\n";
        return;
    }
    require(device.caps().enabled.buffer_device_address,
            "ray query enables device address");
    bool rejected = false;
    try {
        const float none[] = {0.F};
        (void)device.create_triangle_scene(none, {});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "empty ray-query geometry is rejected");

    const float positions[] = {
        0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 1.F, 0.F,
    };
    const std::uint32_t indices[] = {0, 1, 2};
    TriangleSceneOptions options;
    options.geometry_flags = 0;
    auto scene = device.create_triangle_scene(positions, indices, options);
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
    auto bottom = device.create_bottom_level(positions, indices, options);
    std::weak_ptr<const BottomLevel> weak = bottom;
    std::vector<TriangleInstance> instances(2);
    instances[0].geometry = bottom;
    instances[0].custom_index = 9;
    instances[1].geometry = bottom;
    instances[1].custom_index = 19;
    instances[1].transform.matrix[0][3] = 2;
    auto shared_scene = device.create_instance_scene(instances);
    bottom.reset();
    instances.clear();
    require(!weak.expired(), "TLAS must retain shared BLAS ownership");
    require(
        trace_ray(device, shared_scene, pipeline, hits, 2.2F, .2F, 1, 0, 0, -1) == 1,
        "translated shared BLAS instance should be hit");
    shared_scene = TriangleScene{};
    require(weak.expired(), "BLAS should be released when its last TLAS is destroyed");
    require(device.validation_error_count() == 0, "ray query validation errors");
    std::cout << "triangle scene: alpha candidates, typed descriptors, push="
              << push_descriptors << "\n";
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
    auto view = device.create_image_view(image.handle(), image.format(),
                                         VK_IMAGE_ASPECT_COLOR_BIT);
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
    vkCmdCopyImageToBuffer(command, image.handle(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle(), 1,
                           &region);
    encoder.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
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
        test_timestamp_pool();
        test_compute_add();
        test_sampled_image(true);
        test_sampled_image(false);
        test_pipeline_cache(true);
        test_pipeline_cache(false);
        test_resource_device_lifetime();
        test_dispatch_indirect();
        test_encoder_ring();
        test_adopt();
        test_triangle_scene(true);
        test_triangle_scene(false);
        test_graphics_triangle();
    } catch (const std::exception &error) {
        std::cerr << "photara_vk runtime test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "photara_vk runtime test: ok\n";
    return 0;
}
