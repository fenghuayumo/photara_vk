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
- `ComputePipeline` from SPIR-V, optionally using a device-compatible `PipelineCache`
- `CommandEncoder`: `dispatch`, `dispatch_indirect`, `fill_u32`, `copy`,
  `submit`, `submit_wait`
- `CommandEncoder::prepare_descriptors` creates immutable `ComputeDescriptors`
  for ordinary non-push descriptor pipelines. Both dispatch APIs accept the
  prepared set across submissions and pipelines with identical descriptor
  types/counts. Preparing snapshots image-array entries; changing the source
  binding array later does not update a prepared set. Create a new set when
  resources change. Referenced buffers, views, samplers and acceleration
  structures must outlive GPU use; encoders retain the set/pool through retirement.
- `CommandEncoder::statistics()` reports actual direct/indirect dispatch,
  queue-submission and fence-wait API counts over the encoder's lifetime, plus
  descriptor allocations/updates/binds and pipeline binds. Empty submits/waits
  do not count. Zero-group indirect commands do count. A fence-wait call need
  not block. Raw Vulkan calls through `native()` are excluded; obtaining the
  native command buffer invalidates cached binding state.
- `TimestampPool` and encoder timestamp/reset commands, with queue timestamp
  period and valid-bits reporting
- `BarrierPolicy`: `none`, `after_compute`, `after_compute_indirect`
- `EncoderRing` for overlapping command buffers
- `DescriptorBinding` / `binding(scene)` for mixed storage-buffer and acceleration-
  structure descriptors, through push descriptors or allocated descriptor sets.
  `BufferBinding` remains a source-compatible alias.
- Sampled `ImageView` and device-owned `Sampler` descriptors work through both
  push descriptors and ordinary sets. `transition_image` and `copy_to_image`
  support explicit color-image uploads (base 2D level, tightly packed supported
  formats); callers keep the Image alive through its views and GPU submission.
  `supports_linear_sampling` reports format support. The sampler uses bilinear
  clamp-to-edge filtering at level zero. No implicit layout tracking is performed.
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

For a pipeline created with push descriptors disabled, prepare once and reuse:

```cpp
auto descriptors = encoder.prepare_descriptors(pipeline, resources);
encoder.dispatch(pipeline, descriptors, &first_constants, sizeof(first_constants), groups);
encoder.dispatch(pipeline, descriptors, &next_constants, sizeof(next_constants), groups);
encoder.submit_wait();
encoder.dispatch(pipeline, descriptors, &last_constants, sizeof(last_constants), groups);
encoder.submit_wait();
auto counts = encoder.statistics(); // 3 dispatches, 2 submissions, 2 fence waits, 1 set update
```

The caller must keep Buffer/Pipeline resources alive until the encoder finishes.
All resources created by `Device` factories (buffers, pipelines, timestamp pools,
images/views, render passes/framebuffers, caches and acceleration structures) retain
the device implementation, including across moves. Low-level constructors taking
raw Vulkan handles borrow the device; the host must keep it alive. `Device::adopt`
also borrows the external device: retaining its wrapper does not transfer ownership
of the host's `VkDevice`/`VkInstance`. Image views/framebuffers and graphics pipelines
still borrow any image, attachment, render-pass or layout handles supplied by callers. Build
scratch addresses are aligned to the reported device requirement. AS build writes
are made visible to subsequent builds and compute ray queries. The convenience
`create_triangle_scene` API still creates one mesh and one identity instance.

## Build

Requires CMake 3.24+, a C++20 compiler, and Vulkan SDK 1.2+. Tests require DXC.

```powershell
cmake -S . -B build -DPHOTARA_BUILD_TESTS=ON
cmake --build build --config Release --target photara_vk --target photara_vk_test
```

Use `add_subdirectory` and link `photara_vk::photara_vk`, or install the standalone
package (Vulkan SDK and a C++20 compiler suffice when tests are disabled):

```powershell
cmake -S . -B build -DPHOTARA_BUILD_TESTS=OFF
cmake --build build --config Release
cmake --install build --config Release --component photara_vk --prefix D:/sdk/photara_vk
```

An independent consumer sets `CMAKE_PREFIX_PATH` to the installation prefix:

```cmake
find_package(photara_vk 0.3 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE photara_vk::photara_vk)
```

The installed package includes `photara_vk_embed_hlsl`; only calling that helper
requires DXC. The `photara.vk.package` CTest installs and relocates the package,
then builds/runs a separate consumer. No renderer or Python target is needed.

## Pipeline cache

```cpp
auto cache = device.create_pipeline_cache();
auto pipeline = device.create_compute(spirv, descriptor_types, push_bytes, &cache);
auto bytes = cache.data(); // host chooses where/when to persist
// On a later run, on a compatible GPU/driver:
auto restored = device.create_pipeline_cache(bytes);
auto next = device.create_compute(spirv, descriptor_types, push_bytes, &restored);
```

An empty byte span creates an empty cache. Nonempty input must be unmodified bytes
previously returned by `data()`: truncated or incompatible Vulkan headers throw
`std::invalid_argument` before reaching the driver. Device/vendor IDs and the driver
cache UUID must match. Header validation is not a checksum for the opaque payload.
The host can catch rejection and explicitly create an empty cache. A cache from a
different logical device, or an empty/moved-from cache passed to pipeline creation,
is rejected. Cache-backed compute creation and serialization share a mutex;
callers must synchronize moving/destroying the cache object itself. Pipelines can
outlive the cache. This does not guarantee a speedup or imply an on-disk cache in
the host renderer. Graphics pipeline caching is not implemented.

`Capabilities::max_storage_buffer_range` exposes the descriptor range limit, so
compute adapters do not need to query physical-device properties directly.

## Remote

https://github.com/fenghuayumo/photara_vk.git
