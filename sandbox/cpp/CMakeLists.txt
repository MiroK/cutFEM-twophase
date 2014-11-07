# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME run.x)
project(${PROJECT_NAME})

# Set verbose output while testing CMake
set(CMAKE_VERBOSE_MAKEFILE 1)

# Set CMake behavior
cmake_policy(SET CMP0004 OLD)

# Get CUTFEM configuration data (cutfem-config.cmake must be in CUTFEM_CMAKE_CONFIG_PATH)
#FIXME: DOLFIN should not be included
find_package(DOLFIN)
find_package(cutfem)

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Compiler definitions
add_definitions(${CUTFEM_CXX_DEFINITIONS})

# Add special CUTFEM compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUTFEM_CXX_FLAGS}")

# Include directories
include_directories(${CUTFEM_INCLUDE_DIRS})
include_directories(SYSTEM ${CUTFEM_3RD_PARTY_INCLUDE_DIRS})

# Executable
add_executable(${PROJECT_NAME} main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} ${CUTFEM_LIBRARIES} ${CUTFEM_3RD_PARTY_LIBRARIES})
message(STATUS "CUTFEM_3RD_PARTY_LIBRARIES are ${CUTFEM_3RD_PARTY_LIBRARIES}") 
