cmake_minimum_required(VERSION 3.15)
project(cbr_control VERSION 0.1.0)

include(CTest)
include(GoogleTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(autodiff REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(osqp REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED)
find_package(Ceres REQUIRED)
find_package(cbr_utils REQUIRED)
find_package(cbr_math REQUIRED)

option(BUILD_EXAMPLES "Build examples." OFF)
option(BUILD_TESTING "Build tests." ON)

# ---------------------------------------------------------------------------------------
# TARGETS
# ---------------------------------------------------------------------------------------

add_library(osqp-cpp SHARED
  src/osqp-cpp.cpp
)
target_include_directories(osqp-cpp
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(osqp-cpp
PUBLIC
  osqp::osqp
  Eigen3::Eigen
)

add_library(pge SHARED
  src/ceres_marginalization.cpp
)
target_include_directories(pge
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(pge
PUBLIC
  ${CERES_LIBRARIES}
  cbr_math::cbr_math
  cbr_utils::cbr_utils
)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}
INTERFACE
  osqp-cpp
  autodiff::autodiff
  Eigen3::Eigen
  Sophus::Sophus
  cbr_math::cbr_math
  cbr_utils::cbr_utils
)

# ---------------------------------------------------------------------------------------
# INSTALLATION
# ---------------------------------------------------------------------------------------

include(GNUInstallDirs)

# Header files
install(
  DIRECTORY    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
  DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}
)

# Targets
install(
  TARGETS               osqp-cpp ${PROJECT_NAME}
  EXPORT                ${PROJECT_NAME}_targets
  LIBRARY DESTINATION   ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION   ${CMAKE_INSTALL_LIBDIR}
)


# ---------------------------------------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------------------------------------
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------------------
# CMAKE CONFIGURATION
# ---------------------------------------------------------------------------------------

set(export_dest_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Generate TARGETS files
install(
  EXPORT       ${PROJECT_NAME}_targets
  FILE         ${PROJECT_NAME}Targets.cmake
  NAMESPACE    ${PROJECT_NAME}::
  DESTINATION  ${export_dest_dir}
)

# Generate CONFIG file
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/ConfigTemplate.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${export_dest_dir}
)

# Generate VERSION file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Version.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install CONFIG and VERSION files
install(
  FILES        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Version.cmake
  DESTINATION  ${export_dest_dir}
)

# Custom uninstall target
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
  IMMEDIATE @ONLY
)
add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
)

# ---------------------------------------------------------------------------------------
# TESTING
# ---------------------------------------------------------------------------------------

if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  add_compile_options(-Wall -Wextra -Wpedantic -Werror)

  add_executable(${PROJECT_NAME}_test_asif test/test_asif.cpp)
  target_link_libraries(${PROJECT_NAME}_test_asif PRIVATE ${PROJECT_NAME} osqp-cpp GTest::Main)
  gtest_discover_tests(${PROJECT_NAME}_test_asif)

  add_executable(${PROJECT_NAME}_test_care test/test_care.cpp)
  target_link_libraries(${PROJECT_NAME}_test_care PRIVATE ${PROJECT_NAME} osqp-cpp GTest::Main)
  gtest_discover_tests(${PROJECT_NAME}_test_care)

  add_executable(${PROJECT_NAME}_test_mpc test/test_mpc.cpp)
  target_link_libraries(${PROJECT_NAME}_test_mpc PRIVATE ${PROJECT_NAME} osqp-cpp GTest::Main)
  gtest_discover_tests(${PROJECT_NAME}_test_mpc)

  add_executable(${PROJECT_NAME}_test_pge test/test_pge.cpp)
  target_link_libraries(${PROJECT_NAME}_test_pge PRIVATE ${PROJECT_NAME} pge GTest::Main)
  gtest_discover_tests(${PROJECT_NAME}_test_pge)
endif()

