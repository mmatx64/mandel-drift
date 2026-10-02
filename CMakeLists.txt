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
# Release source archives include the exact SDL source used by the app.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/SDL/CMakeLists.txt")
  set(FETCHCONTENT_SOURCE_DIR_SDL3 "${CMAKE_CURRENT_SOURCE_DIR}/third_party/SDL")
endif()
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
  src/music_travel.h
  src/music_reactivity.h
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
install(FILES README.md LICENSE CUDA-EXCEPTION.txt CUDA-LICENSE.txt DESTINATION .)
install(FILES "${sdl3_SOURCE_DIR}/LICENSE.txt" DESTINATION . RENAME SDL3-LICENSE.txt)

