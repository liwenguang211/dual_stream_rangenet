cmake_minimum_required(VERSION 3.16)
project(DualStreamRangeNet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ────────────────────────────────────────────────────────────
# 构建类型与编译优化
# ────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -funroll-loops")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -march=native")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DDEBUG -fsanitize=address")
endif()

# ────────────────────────────────────────────────────────────
# ONNX Runtime (已安装到 /opt/onnxruntime-linux-x64-1.23.2)
# ────────────────────────────────────────────────────────────
set(ONNXRUNTIME_ROOT "/opt/onnxruntime-linux-x64-1.23.2"
    CACHE PATH "ONNX Runtime 安装目录")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime 头文件未找到: ${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h\n"
        "请通过 -DONNXRUNTIME_ROOT=<path> 指定正确路径")
endif()

# 头文件
include_directories(${ONNXRUNTIME_ROOT}/include)

# 链接目录 & 库
link_directories(${ONNXRUNTIME_ROOT}/lib)

# ────────────────────────────────────────────────────────────
# OpenMP (预处理并行)
# ────────────────────────────────────────────────────────────
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP 已启用")
endif()

# ────────────────────────────────────────────────────────────
# 可选: OpenCV (可视化调试)
# ────────────────────────────────────────────────────────────
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} 已找到")
    add_definitions(-DHAVE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# ────────────────────────────────────────────────────────────
# 包含目录
# ────────────────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)

# ────────────────────────────────────────────────────────────
# 源文件
# ────────────────────────────────────────────────────────────
set(SOURCES
    src/RangeNetInferencer.cpp
    src/utils.cpp
)

# ────────────────────────────────────────────────────────────
# 可执行文件
# ────────────────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(rangenet_inference src/main.cpp ${SOURCES})

# ── 链接库 ───────────────────────────────────────────────────
target_link_libraries(rangenet_inference
    # ★ ONNX Runtime 共享库 (必须链接)
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so

    # 系统库
    pthread
    dl
    m
)

if(OpenCV_FOUND)
    target_link_libraries(rangenet_inference ${OpenCV_LIBS})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(rangenet_inference OpenMP::OpenMP_CXX)
    target_compile_options(rangenet_inference PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# ── RPATH (运行时能找到 libonnxruntime.so) ───────────────────
set_target_properties(rangenet_inference PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# ────────────────────────────────────────────────────────────
# 安装
# ────────────────────────────────────────────────────────────
install(TARGETS rangenet_inference DESTINATION bin)
install(FILES
    include/RangeNetInferencer.hpp
    include/model_config.h
    DESTINATION include
)

# ────────────────────────────────────────────────────────────
# 配置摘要
# ────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== Dual-Stream RangeNet Lite 构建配置 ===")
message(STATUS "  构建类型:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  ORT 路径:    ${ONNXRUNTIME_ROOT}")
message(STATUS "  OpenMP:      ${OpenMP_CXX_FOUND}")
message(STATUS "  OpenCV:      ${OpenCV_FOUND}")
message(STATUS "")
message(STATUS "构建命令:")
message(STATUS "  mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)")
message(STATUS "")
message(STATUS "运行示例:")
message(STATUS "  ./build/bin/rangenet_inference --model models/dual_rangenet_lite.onnx --benchmark")
message(STATUS "  ./build/bin/rangenet_inference --model models/dual_rangenet_lite.onnx --input data/scan.bin")
message(STATUS "")
