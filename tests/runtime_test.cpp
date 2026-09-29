#include "photara_vk/photara_vk.hpp"

#include "add.hlsl.embedded.hpp"

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
    } catch (const std::exception& error) {
        std::cerr << "photara_vk runtime test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "photara_vk runtime test: ok\n";
    return 0;
}
