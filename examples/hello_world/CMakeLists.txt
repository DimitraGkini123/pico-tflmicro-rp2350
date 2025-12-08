
cmake_minimum_required(VERSION 3.12)

project(hello_world C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)


add_executable(hello_world_test "")

target_include_directories(hello_world_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  hello_world_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(hello_world_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/hello_world_float_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/hello_world_int8_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/hello_world_test.cpp
)

target_link_libraries(
  hello_world_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_enable_stdio_usb(hello_world_test 1)
pico_enable_stdio_uart(hello_world_test 0)

pico_add_extra_outputs(hello_world_test)


add_executable(hello_world "")

target_include_directories(hello_world
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  hello_world
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(hello_world
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/constants.cpp
  ${CMAKE_CURRENT_LIST_DIR}/hello_world_float_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/hello_world_int8_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/rp2/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
  ${CMAKE_CURRENT_LIST_DIR}/mlp_int8_model_data.cpp

)

target_link_libraries(
  hello_world
  pico-tflmicro
  hardware_pwm
)

pico_enable_stdio_usb(hello_world 1)
pico_enable_stdio_uart(hello_world 0)

pico_add_extra_outputs(hello_world)


add_executable(output_handler_test "")

target_include_directories(output_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  output_handler_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(output_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/output_handler_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/rp2/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
)

target_link_libraries(
  output_handler_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(output_handler_test)
add_executable(fall_detection
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fall_detection_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/svm_model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.cpp
)

target_link_libraries(fall_detection
  pico_stdlib      # μόνο αυτό χρειάζεται για τώρα
)
pico_enable_stdio_usb(fall_detection 1)
pico_enable_stdio_uart(fall_detection 0)

pico_add_extra_outputs(fall_detection)


set(PICO_STACK_SIZE 4096)
add_executable(seizure_detection "")

target_include_directories(seizure_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  seizure_detection
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(seizure_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/constants.cpp
  ${CMAKE_CURRENT_LIST_DIR}/seizure_mlp_int8_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/seizure_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/rp2/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h

)

target_link_libraries(
  seizure_detection
  pico-tflmicro
  hardware_pwm
)

pico_enable_stdio_usb(seizure_detection 1)
pico_enable_stdio_uart(seizure_detection 0)

pico_add_extra_outputs(seizure_detection)

add_executable(glucose_detection
    ${CMAKE_CURRENT_LIST_DIR}/glucose_main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlp_glucose_int8_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/constants.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rp2/output_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/constants.h
    ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
)

target_include_directories(glucose_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  glucose_detection
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_link_libraries(
  glucose_detection
  pico-tflmicro
  hardware_pwm
)

pico_enable_stdio_usb(glucose_detection 1)
pico_enable_stdio_uart(glucose_detection 0)

pico_add_extra_outputs(glucose_detection)