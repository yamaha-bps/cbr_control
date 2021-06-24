include(FetchContent)

fetchcontent_declare(matplotplusplus
  GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
  GIT_SHALLOW    ON
  GIT_TAG        origin/master
)

fetchcontent_getproperties(matplotplusplus)
if(NOT ${matplotplusplus_POPULATED})
  fetchcontent_populate(matplotplusplus)
  add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable(pendulum pendulum_main.cpp)
target_link_libraries(pendulum ${PROJECT_NAME} matplot)

add_executable(segway segway_main.cpp)
target_link_libraries(segway ${PROJECT_NAME} matplot)

add_executable(so3 so3_main.cpp)
target_link_libraries(so3 ${PROJECT_NAME} matplot)

add_executable(se2 se2_main.cpp)
target_link_libraries(se2 ${PROJECT_NAME} matplot)

add_executable(asif_integrator asif_integrator.cpp)
target_link_libraries(asif_integrator ${PROJECT_NAME} matplot)

