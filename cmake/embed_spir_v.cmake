if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE OR NOT DEFINED VARIABLE_NAME)
    message(FATAL_ERROR "INPUT_FILE, OUTPUT_FILE, and VARIABLE_NAME are required")
endif()

file(READ "${INPUT_FILE}" shader_hex HEX)
# Convert in bulk: per-byte substring/appends become quadratic for large
# shaders. Break the hex into 16-byte lines before formatting each byte.
string(REGEX REPLACE "(................................)" "\\1\n    " shader_lines "${shader_hex}")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," shader_bytes "${shader_lines}")
file(WRITE "${OUTPUT_FILE}"
    "#pragma once\n\ninline constexpr unsigned char ${VARIABLE_NAME}[] = {\n    ${shader_bytes}\n};\n")
