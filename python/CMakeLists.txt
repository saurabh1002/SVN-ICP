cmake_minimum_required(VERSION 3.18)
project(pysvnicp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Python ---------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# ---- PyTorch --------------------------------------------------------------
# TORCH_DIR may be passed explicitly (e.g. -DTORCH_DIR=/opt/libtorch/share/cmake/Torch).
# If omitted, we try to infer it from the active Python interpreter so that
# the Torch version that matches the installed `torch` Python package is used.
if(NOT DEFINED TORCH_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_detect_result
    )
    if(NOT _torch_detect_result EQUAL 0)
        message(FATAL_ERROR
            "Could not detect TORCH_DIR automatically. "
            "Either activate the Python environment that has torch installed, "
            "or pass -DTORCH_DIR=/path/to/libtorch/share/cmake/Torch to cmake.")
    endif()
endif()
find_package(Torch REQUIRED HINTS "${TORCH_DIR}")

# ---- pybind11 -------------------------------------------------------------
# Prefer the pybind11 that ships with the active Python / pip environment.
if(NOT DEFINED pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_detect_result
    )
    if(NOT _pybind11_detect_result EQUAL 0)
        message(WARNING
            "Could not detect pybind11 via Python. "
            "Falling back to system find_package. "
            "Install pybind11 with: pip install pybind11")
        unset(pybind11_DIR)
    endif()
endif()
find_package(pybind11 REQUIRED HINTS "${pybind11_DIR}")

# ---- Other dependencies ---------------------------------------------------
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)
find_package(GTSAMCMakeTools)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(tsl-robin-map REQUIRED)

# Path to the svn-icp source tree (sibling directory of this python/ directory)
set(SVNICP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../svn-icp")

# ---- Core shared library (ROS-free) ---------------------------------------
add_library(svnicp_core SHARED
    ${SVNICP_SRC}/src/core/SVGDICP.cpp
    ${SVNICP_SRC}/src/core/SVNICP.cpp
    ${SVNICP_SRC}/src/core/ICPUtils.cpp
    ${SVNICP_SRC}/src/core/knn/knn.cu
    ${SVNICP_SRC}/src/core/knn/knn_cpu.cpp
    ${SVNICP_SRC}/src/core/knn/knn.cpp
    ${SVNICP_SRC}/src/core/VoxelHashMap.cpp
)

# Expose SVNICP_NO_ROS to all consumers so that DataTypes.h elides rclcpp
target_compile_definitions(svnicp_core PUBLIC SVNICP_NO_ROS)

set_property(TARGET svnicp_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(svnicp_core PUBLIC
    ${SVNICP_SRC}/include
    ${TORCH_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${TBB_INCLUDE_DIR}
    ${tsl-robin-map_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

target_link_libraries(svnicp_core PUBLIC
    ${TORCH_LIBRARIES}
    ${PCL_LIBRARIES}
    gtsam
    gtsam_unstable
    TBB::tbb
    Python3::Python
)

# ---- Python extension module ----------------------------------------------
pybind11_add_module(pysvnicp MODULE bindings.cpp)

target_compile_definitions(pysvnicp PRIVATE SVNICP_NO_ROS)

target_link_libraries(pysvnicp PRIVATE
    svnicp_core
    ${TORCH_LIBRARIES}
    ${PCL_LIBRARIES}
    pybind11::module
    Python3::Python
)

# Print install location hint
install(TARGETS pysvnicp
    LIBRARY DESTINATION ${Python3_SITEARCH}
)
message(STATUS
    "pysvnicp will be installed to: ${Python3_SITEARCH}. "
    "Alternatively, copy/symlink the built .so to your project directory.")
