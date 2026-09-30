# Compile HLSL compute shaders with DXC and embed the SPIR-V as a C++ header.
#
#   photara_vk_embed_hlsl(
#       TARGET my_target
#       SHADERS add.hlsl other.hlsl
#       [PROFILE cs_6_0]
#       [ENV vulkan1.2]
#       [INCLUDE_DIR path]
#       [OUTPUT_DIR path]
#       [DEPENDS extra.hlsli])
#
# The generated headers are added as sources of TARGET and OUTPUT_DIR is added
# to its private include path. VARIABLE_NAME is the C identifier of the file
# name plus _spv, e.g. add.hlsl -> add_hlsl_spv.

set(PHOTARA_VK_EMBED_SPIR_V "${CMAKE_CURRENT_LIST_DIR}/embed_spir_v.cmake")

function(photara_vk_embed_hlsl)
    cmake_parse_arguments(ARG "" "TARGET;PROFILE;ENV;INCLUDE_DIR;OUTPUT_DIR" "SHADERS;DEPENDS;EXTRA" ${ARGN})
    if(NOT ARG_TARGET)
        message(FATAL_ERROR "photara_vk_embed_hlsl: TARGET is required")
    endif()
    if(NOT ARG_SHADERS)
        message(FATAL_ERROR "photara_vk_embed_hlsl: SHADERS is required")
    endif()
    if(NOT PHOTARA_VK_DXC)
        message(FATAL_ERROR "photara_vk_embed_hlsl: dxc was not found")
    endif()
    if(NOT ARG_PROFILE)
        set(ARG_PROFILE cs_6_0)
    endif()
    if(NOT ARG_ENV)
        set(ARG_ENV vulkan1.2)
    endif()
    if(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/photara_vk_shaders")
    endif()
    if(NOT ARG_INCLUDE_DIR)
        set(ARG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    set(embedded_headers)
    foreach(shader IN LISTS ARG_SHADERS)
        if(IS_ABSOLUTE "${shader}")
            set(shader_source "${shader}")
        else()
            set(shader_source "${CMAKE_CURRENT_SOURCE_DIR}/${shader}")
        endif()
        get_filename_component(shader_name "${shader}" NAME)
        set(shader_binary "${ARG_OUTPUT_DIR}/${shader_name}.spv")
        set(shader_header "${ARG_OUTPUT_DIR}/${shader_name}.embedded.hpp")
        string(MAKE_C_IDENTIFIER "${shader_name}_spv" shader_variable)
        add_custom_command(
            OUTPUT "${shader_binary}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${ARG_OUTPUT_DIR}"
            COMMAND "${PHOTARA_VK_DXC}"
                    -spirv -T ${ARG_PROFILE} -E main -O3
                    -fspv-target-env=${ARG_ENV}
                    -fvk-use-dx-layout
                    ${ARG_EXTRA}
                    -I "${ARG_INCLUDE_DIR}"
                    "${shader_source}"
                    -Fo "${shader_binary}"
            DEPENDS "${shader_source}" ${ARG_DEPENDS}
            VERBATIM)
        add_custom_command(
            OUTPUT "${shader_header}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${ARG_OUTPUT_DIR}"
            COMMAND "${CMAKE_COMMAND}"
                    -DINPUT_FILE=${shader_binary}
                    -DOUTPUT_FILE=${shader_header}
                    -DVARIABLE_NAME=${shader_variable}
                    -P "${PHOTARA_VK_EMBED_SPIR_V}"
            DEPENDS "${shader_binary}" "${PHOTARA_VK_EMBED_SPIR_V}"
            VERBATIM)
        list(APPEND embedded_headers "${shader_header}")
    endforeach()

    target_sources(${ARG_TARGET} PRIVATE ${embedded_headers})
    target_include_directories(${ARG_TARGET} PRIVATE "${ARG_OUTPUT_DIR}")
endfunction()
