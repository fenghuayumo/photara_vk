# photara_vk

C++20 Vulkan 1.2 compute runtime. Callers create or adopt a `Device`, then
record work with `Buffer`, `ComputePipeline`, `CommandEncoder`, and
`EncoderRing`.

Windowing, swapchain, and ImGui stay in the host application.

## API

- `Device::create` / `Device::adopt`, with optional features such as
  `push_descriptors`, `buffer_atomic_f32`, and ray query
- `Buffer` with `device_local`, `host_visible`, `host_cached`, and
  `host_visible_device_local`
- `ComputePipeline` from SPIR-V
- `CommandEncoder`: `dispatch`, `dispatch_indirect`, `fill_u32`, `copy`,
  `submit`, `submit_wait`
- `TimestampPool` and encoder timestamp/reset commands, with queue timestamp
  period and valid-bits reporting
- `BarrierPolicy`: `none`, `after_compute`, `after_compute_indirect`
- `EncoderRing` for overlapping command buffers
- `DescriptorBinding` / `binding(scene)` for mixed storage-buffer and acceleration-
  structure descriptors, through push descriptors or allocated descriptor sets.
  `BufferBinding` remains a source-compatible alias.
- `Device::create_bottom_level` creates shared object-space triangle BLASes;
  `Device::create_instance_scene` builds a TLAS from transformed `TriangleInstance`
  records, including custom index and visibility mask. The TLAS retains its BLASes.
- `TriangleSceneOptions` exposes geometry/build flags; geometry flags `0` allows
  non-opaque triangle candidates for alpha testing in the shader.
- `DeviceRequest::synchronization_validation` together with `validation` enables
  synchronization checks; `Device::validation_error_count()` supports test failures.

## Shared geometry example

```cpp
photara::vk::DeviceRequest request;
request.want.ray_query = true;
auto device = photara::vk::Device::create(request);
photara::vk::TriangleSceneOptions options;
options.geometry_flags = 0;
auto geometry = device.create_bottom_level(positions, indices, options);
std::vector<photara::vk::TriangleInstance> objects(2);
objects[0].geometry = objects[1].geometry = geometry;
objects[1].transform.matrix[0][3] = 2.0f;
objects[1].custom_index = 12;
auto scene = device.create_instance_scene(objects);
const photara::vk::DescriptorBinding resources[] = {
    photara::vk::binding(scene), photara::vk::binding(output_buffer)
};
encoder.dispatch(pipeline, resources, &constants, sizeof(constants), groups);
```

The caller must keep Buffer/Pipeline resources alive until the encoder finishes.
An acceleration structure and its BLAS geometry retain the owning device. Build
scratch addresses are aligned to the reported device requirement. AS build writes
are made visible to subsequent builds and compute ray queries. The convenience
`create_triangle_scene` API still creates one mesh and one identity instance.

## Build

Requires CMake 3.24+, a C++20 compiler, and Vulkan SDK 1.2+. Tests require DXC.

```powershell
cmake -S . -B build -DPHOTARA_BUILD_TESTS=ON
cmake --build build --config Release --target photara_vk --target photara_vk_test
```

Inside Photara Studio, `PHOTARA_ENABLE_VK` adds this tree when Vulkan 1.2 is
found.

## Remote

https://github.com/fenghuayumo/photara_vk.git
