cmake_minimum_required(VERSION 3.28)
project(MandelDrift VERSION 0.8.0 LANGUAGES CXX CUDA RC)

# Match the statically linked CUDA runtime and keep the release self-contained.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(NOT WIN32)
  message(FATAL_ERROR "Mandel Drift currently targets Windows.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

include(FetchContent)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG release-3.4.16
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(SDL3)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(MandelDrift WIN32
  src/main.cpp
  src/mandelbrot.cu
  src/mandelbrot.h
  src/settings_overlay.cpp
  src/settings_overlay.h
  src/app_settings.cpp
  src/app_settings.h
  src/bloom_shaders.h
  src/app.rc
  src/ambient_music.cpp
  src/ambient_music.h
  src/ambient_synth.cpp
  src/ambient_synth.h
)
target_include_directories(MandelDrift PRIVATE src)
target_link_libraries(MandelDrift PRIVATE SDL3::SDL3-static CUDA::cudart_static OpenGL::GL gdi32)
set_target_properties(MandelDrift PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
if(MSVC)
  target_compile_options(MandelDrift PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
endif()

install(TARGETS MandelDrift RUNTIME DESTINATION .)
install(FILES README.md DESTINATION .)
install(FILES "${sdl3_SOURCE_DIR}/LICENSE.txt" DESTINATION . RENAME SDL3-LICENSE.txt)

include(CTest)
if(BUILD_TESTING)
  add_executable(julia_test tests/julia_test.cpp src/mandelbrot.cu)
  target_include_directories(julia_test PRIVATE src)
  target_link_libraries(julia_test PRIVATE CUDA::cudart_static)
  set_target_properties(julia_test PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  add_test(NAME julia_numerics COMMAND julia_test)
  set_tests_properties(julia_numerics PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(trip_effects_test tests/trip_effects_test.cpp)
  target_include_directories(trip_effects_test PRIVATE src)
  add_test(NAME trip_effects COMMAND trip_effects_test)
  add_executable(mandelbrot_benchmark tests/mandelbrot_benchmark.cpp
    tests/mandelbrot_baseline.cu src/mandelbrot.cu)
  target_include_directories(mandelbrot_benchmark PRIVATE src)
  target_link_libraries(mandelbrot_benchmark PRIVATE CUDA::cudart_static)
  set_target_properties(mandelbrot_benchmark PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  add_test(NAME mandelbrot_numerics COMMAND mandelbrot_benchmark)
  set_tests_properties(mandelbrot_numerics PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(frame_budget_test tests/frame_budget_test.cpp)
  target_include_directories(frame_budget_test PRIVATE src)
  add_test(NAME frame_budget COMMAND frame_budget_test)
  add_executable(settings_test tests/settings_test.cpp src/app_settings.cpp)
  target_include_directories(settings_test PRIVATE src)
  add_test(NAME settings_round_trip COMMAND settings_test)
  add_executable(music_test tests/music_test.cpp src/ambient_synth.cpp src/ambient_music.cpp)
  target_include_directories(music_test PRIVATE src)
  target_link_libraries(music_test PRIVATE SDL3::SDL3-static)
  add_test(NAME ambient_music COMMAND music_test)
endif()

