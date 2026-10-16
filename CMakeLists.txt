cmake_minimum_required(VERSION 3.24)

if(NOT TARGET photara_vk)
    project(photara_vk VERSION 0.3.0 LANGUAGES CXX)

    find_package(Vulkan 1.2 REQUIRED)
    find_package(Threads REQUIRED)

    add_library(photara_vk STATIC
        src/device.cpp
        src/buffer.cpp
        src/pipeline.cpp
        src/command.cpp
        src/accel.cpp
        src/graphics.cpp
        src/query.cpp
        src/cache.cpp)
    add_library(photara_vk::photara_vk ALIAS photara_vk)

    target_include_directories(photara_vk
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(photara_vk PUBLIC cxx_std_20)
    target_link_libraries(photara_vk PUBLIC Vulkan::Vulkan Threads::Threads)
    set_target_properties(photara_vk PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        FOLDER "photara_vk")
    if(MSVC)
        target_compile_options(photara_vk PRIVATE /W4 /permissive- /EHsc /utf-8)
    else()
        target_compile_options(photara_vk PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(PHOTARA_BUILD_TESTS OR BUILD_TESTING)
    enable_testing()
endif()

find_program(PHOTARA_VK_DXC
    NAMES dxc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
set(PHOTARA_VK_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE INTERNAL
    "Directory containing PhotaraVkShaders.cmake")
include("${PHOTARA_VK_CMAKE_DIR}/PhotaraVkShaders.cmake")

if((PHOTARA_BUILD_TESTS OR BUILD_TESTING) AND PHOTARA_VK_DXC)
    add_executable(photara_vk_test tests/runtime_test.cpp)
    photara_vk_embed_hlsl(
        TARGET photara_vk_test
        SHADERS tests/shaders/add.hlsl
        tests/shaders/sample_image.hlsl
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

# Relocatable standalone package; no renderer, Python, shader compiler or window dependency.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(photara_vk_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/photara_vk")
install(TARGETS photara_vk EXPORT photara_vkTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT photara_vk)
install(DIRECTORY include/photara_vk DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT photara_vk)
install(EXPORT photara_vkTargets NAMESPACE photara_vk::
    DESTINATION "${photara_vk_INSTALL_CMAKEDIR}" COMPONENT photara_vk)
configure_package_config_file(cmake/photara_vkConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/photara_vkConfig.cmake"
    INSTALL_DESTINATION "${photara_vk_INSTALL_CMAKEDIR}")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/photara_vkConfigVersion.cmake"
    VERSION 0.2.0 COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/photara_vkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/photara_vkConfigVersion.cmake"
    cmake/PhotaraVkShaders.cmake cmake/embed_spir_v.cmake
    DESTINATION "${photara_vk_INSTALL_CMAKEDIR}" COMPONENT photara_vk)
if(PHOTARA_BUILD_TESTS OR BUILD_TESTING)
    add_test(NAME photara.vk.package COMMAND "${CMAKE_COMMAND}"
        "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-test"
        "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer"
        "-DCONFIG=$<CONFIG>"
        "-DGENERATOR=${CMAKE_GENERATOR}"
        "-DPLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DTOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package_smoke.cmake")
endif()
