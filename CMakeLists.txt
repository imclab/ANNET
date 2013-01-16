# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
cmake_minimum_required (VERSION 2.8)
project (ANNetGPGPU)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(CUDAThrust)

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (src/include)
INCLUDE_DIRECTORIES (src/include/base)
INCLUDE_DIRECTORIES (src/include/math)
INCLUDE_DIRECTORIES (src/include/containers)
INCLUDE_DIRECTORIES (src/include/gpgpu)

set( SourceFiles 
  src/2DArray.cpp
  src/3DArray.cpp
  src/AbsLayer.cpp
  src/AbsNet.cpp
  src/AbsNeuron.cpp
  src/BPLayer.cpp
  src/BPNet.cpp
  src/BPNeuron.cpp
  src/Edge.cpp
  src/Functions.cpp
  src/HFLayer.cpp
  src/HFNet.cpp
  src/HFNeuron.cpp
  src/SOMLayer.cpp
  src/SOMNet.cpp
  src/SOMNeuron.cpp
  src/TrainingSet.cpp
)

set( CUDASourceFiles
  src/BPNetGPU.cpp
  src/SOMNetGPU.cpp
  src/BPKernel.cu
  src/SOMKernel.cu
  src/HFKernel.cu
  src/Matrix.cu
)




if (CUDA_FOUND)
  INCLUDE(FindCUDA)
  set(CUDA_NVCC_FLAGS "-arch=sm_13")
  include_directories (${CUDA_SDK_ROOT_DIR}/C/common/inc/)

  if (CUDATHRUST_FOUND)
    include_directories (${CUDATHRUST_INCLUDE_DIR})
  endif (CUDATHRUST_FOUND)
  
  ADD_DEFINITIONS("-DCUDA") # needed for conditional compilation of some files
endif (CUDA_FOUND)

# Create a library called "ANNet" which includes the source files listed in "SourceFiles".
# The extension is already found. Any number of sources could be listed here.
if (BZIP2_FOUND)
  if (CUDA_FOUND)
    cuda_add_library (ANNet SHARED ${SourceFiles} ${CUDASourceFiles} ${BZIP_INCLUDE_DIRS}) 
  elseif (NOT CUDA_FOUND)
    add_library (ANNet SHARED ${SourceFiles} ${BZIP_INCLUDE_DIRS})
  endif(CUDA_FOUND)

  # -fopenmp necessary for mingw NOT gcc
  if(OPENMP_FOUND)
    target_link_libraries (ANNet ${BZIP2_LIBRARIES} -fopenmp)
  elseif(NOT OPENMP_FOUND)
    target_link_libraries (ANNet ${BZIP2_LIBRARIES})
  endif(OPENMP_FOUND)

endif (BZIP2_FOUND)