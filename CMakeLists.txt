cmake_minimum_required(VERSION 3.12)

include(vendor/pico_sdk_import.cmake)

project(pico-neuralclick C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

pico_sdk_init()

add_subdirectory(vendor)

add_executable(neuralclick main.cpp)
target_link_libraries(neuralclick PRIVATE
  hardware_sync
  pico_stdlib
  pico-tflmicro
)
set_target_properties(neuralclick PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(neuralclick 1)
pico_add_extra_outputs(neuralclick)
