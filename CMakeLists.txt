# moe-serve — 纯 C 的 MoE 推理服务
# ============================================
# Phase 1: 核心基础 (Tensor / Mat / Math / MemPool / Logger)
# Phase 2: GGUF 加载 + MoE 推理引擎
# Phase 3: HTTP/REST + WebSocket API (libmicrohttpd)
# Phase 4: 部署与监控

cmake_minimum_required(VERSION 3.15)
project(moe-serve LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)   # gnu11: localtime_r 等 POSIX 函数

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# 构建产物统一放到 output/ 目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

# ---- 可选依赖探测 ----
# OpenMP: 用 -fopenmp 标志（编译+链接）交给活跃编译器解析 libgomp，
# 避免某些多 gcc 版本环境下解析成不存在的绝对库路径(如 gcc-11 的 libgomp.so)。
find_package(OpenMP QUIET)
set(MOE_OPENMP "")
set(MOE_HAS_OPENMP OFF)
if(OpenMP_C_FOUND)
  set(MOE_OPENMP "-fopenmp")
  set(MOE_HAS_OPENMP ON)
  message(STATUS "OpenMP 已启用 (-fopenmp)")
else()
  message(WARNING "未找到 OpenMP，将串行编译（可后续开启并行）")
endif()

function(moe_apply_openmp tgt)
  if(MOE_HAS_OPENMP)
    target_compile_options(${tgt} PRIVATE ${MOE_OPENMP})
    target_link_options(${tgt} PRIVATE ${MOE_OPENMP})
  endif()
endfunction()

# ---- HTTP 服务依赖 (libmicrohttpd / jansson)，默认可选 ----
option(BUILD_HTTP "Build HTTP server support (--serve)" OFF)
set(MOE_HAVE_HTTP OFF)
if(BUILD_HTTP)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(MO_MHD IMPORTED_TARGET libmicrohttpd)
    pkg_check_modules(MO_JANSSON IMPORTED_TARGET jansson)
    if(MO_MHD_FOUND AND MO_JANSSON_FOUND)
      set(MOE_HAVE_HTTP ON)
      message(STATUS "HTTP --serve 可用 (libmicrohttpd ${MO_MHD_VERSION} + jansson)")
    else()
      message(WARNING "缺少 libmicrohttpd/jansson，--serve 不可用（apt-get install libmicrohttpd-dev libjansson-dev）")
    endif()
  else()
    message(WARNING "未找到 pkg-config，--serve 不可用")
  endif()
endif()

# ---- 基础编译选项 ----
set(MOE_WARNINGS -Wall -Wextra -Wpedantic)

# ---- 库：core (tensor + matrix + transformer) ----
add_library(moe_core STATIC
  src/core/tensor.c
  src/core/matrix.c
  src/core/transformer.c)
target_include_directories(moe_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(moe_core PRIVATE ${MOE_WARNINGS} -O3)

# ---- 库：utils (math + mempool + logger) ----
add_library(moe_utils STATIC
  src/utils/math.c
  src/utils/mempool.c
  src/utils/logger.c)
target_include_directories(moe_utils PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(moe_utils PRIVATE ${MOE_WARNINGS} -O3)

# ---- 库：model (config + tokenizer + model/loader + gguf + llama) ----
add_library(moe_model STATIC
  src/model/config.c
  src/model/tokenizer.c
  src/model/model.c
  src/model/gguf.c
  src/model/llama.c
  src/model/llama_tokenizer.c)
target_include_directories(moe_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(moe_model PRIVATE ${MOE_WARNINGS} -O3)

# ---- 库：infer (sample + generate) ----
add_library(moe_infer STATIC
  src/infer/sample.c
  src/infer/generate.c)
target_include_directories(moe_infer PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(moe_infer PRIVATE ${MOE_WARNINGS} -O3)

set(MOE_INTERFACE_LIBS moe_infer moe_model moe_core moe_utils m)
# 静态库存在循环依赖 (moe_model↔moe_infer)，用链接组解决
set(MOE_GROUP "-Wl,--start-group;moe_infer;moe_model;moe_core;moe_utils;-Wl,--end-group;m")

# ---- 可执行文件 ----
add_executable(pmai src/main.c)
target_include_directories(pmai PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(pmai PRIVATE ${MOE_WARNINGS} -O3)
target_link_libraries(pmai PRIVATE ${MOE_GROUP})
moe_apply_openmp(pmai)
if(MOE_HAVE_HTTP)
  target_sources(pmai PRIVATE src/server/api.c)
  target_link_libraries(pmai PRIVATE PkgConfig::MO_MHD PkgConfig::MO_JANSSON)
  target_compile_definitions(pmai PRIVATE MOE_HTTP=1)
endif()

# ---- llama 架构运行器（构建到 output/）----
add_executable(pmai-llama src/llama_main.c)
target_include_directories(pmai-llama PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(pmai-llama PRIVATE ${MOE_WARNINGS} -O3)
target_link_libraries(pmai-llama PRIVATE ${MOE_GROUP})
moe_apply_openmp(pmai-llama)

# ---- 测试 ----
enable_testing()
add_executable(test_core tests/test_core.c)
target_include_directories(test_core PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_core PRIVATE ${MOE_WARNINGS} -O2)
target_link_libraries(test_core PRIVATE ${MOE_GROUP})
moe_apply_openmp(test_core)
add_test(NAME core COMMAND test_core)

# ---- llama 分词器独立测试 ----
add_executable(test_llmtok tests/test_llmtok.c)
target_include_directories(test_llmtok PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_llmtok PRIVATE ${MOE_WARNINGS} -O2)
target_link_libraries(test_llmtok PRIVATE ${MOE_GROUP})
moe_apply_openmp(test_llmtok)
