cmake_minimum_required(VERSION 3.24)

if(NOT TARGET photara_vk)
    project(photara_vk VERSION 0.1.0 LANGUAGES CXX)

    find_package(Vulkan 1.2 REQUIRED)

    add_library(photara_vk STATIC
        src/device.cpp
        src/buffer.cpp
        src/pipeline.cpp
        src/command.cpp
        src/accel.cpp
        src/graphics.cpp)
    add_library(photara_vk::photara_vk ALIAS photara_vk)

    target_include_directories(photara_vk
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(photara_vk PUBLIC cxx_std_20)
    target_link_libraries(photara_vk PUBLIC Vulkan::Vulkan)
    set_target_properties(photara_vk PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        FOLDER "photara_vk")
    if(MSVC)
        target_compile_options(photara_vk PRIVATE /W4 /permissive- /EHsc /utf-8)
    else()
        target_compile_options(photara_vk PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

find_program(PHOTARA_VK_DXC
    NAMES dxc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/PhotaraVkShaders.cmake")

if((PHOTARA_BUILD_TESTS OR BUILD_TESTING) AND PHOTARA_VK_DXC)
    add_executable(photara_vk_test tests/runtime_test.cpp)
    photara_vk_embed_hlsl(
        TARGET photara_vk_test
        SHADERS tests/shaders/add.hlsl
        INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders"
        OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    photara_vk_embed_hlsl(
        TARGET photara_vk_test
        SHADERS tests/shaders/ray_query.hlsl
        PROFILE cs_6_5
        EXTRA -fspv-extension=SPV_KHR_ray_query
        INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders"
        OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    photara_vk_embed_hlsl(
        TARGET photara_vk_test
        SHADERS tests/shaders/triangle.vs.hlsl
        PROFILE vs_6_0
        INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders"
        OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    photara_vk_embed_hlsl(
        TARGET photara_vk_test
        SHADERS tests/shaders/triangle.ps.hlsl
        PROFILE ps_6_0
        INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders"
        OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(photara_vk_test PRIVATE photara_vk::photara_vk)
    target_compile_features(photara_vk_test PRIVATE cxx_std_20)
    set_target_properties(photara_vk_test PROPERTIES FOLDER "photara_vk")
    if(MSVC)
        target_compile_options(photara_vk_test PRIVATE /W4 /permissive- /EHsc /utf-8)
    endif()
    add_test(NAME photara.vk.runtime COMMAND photara_vk_test)
    message(STATUS "photara_vk tests: ON (dxc=${PHOTARA_VK_DXC})")
elseif(PHOTARA_BUILD_TESTS OR BUILD_TESTING)
    message(STATUS "photara_vk tests: skipped (dxc not found)")
endif()
