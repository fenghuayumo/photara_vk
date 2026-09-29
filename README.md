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
- `BarrierPolicy`: `none`, `after_compute`, `after_compute_indirect`
- `EncoderRing` for overlapping command buffers

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
