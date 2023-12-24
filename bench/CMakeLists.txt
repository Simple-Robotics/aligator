#
# Copyright (C) 2023 LAAS-CNRS, INRIA
#

find_package(benchmark REQUIRED)

# Create a benchmark
function(create_bench exfile croc)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "bench-${exname}")
  create_ex_or_bench(${exfile} ${exname})

  target_link_libraries(${exname} PRIVATE benchmark::benchmark)
  if(croc)
    target_link_libraries(${exname} PUBLIC aligator::croc_compat croc_talos_arm_utils)
  endif(croc)
endfunction()

create_bench("lqr.cpp" FALSE)
if(BUILD_CROCODDYL_COMPAT)
  # create an utility library to avoid recompiling crocoddyl talos arm problem
  add_library(croc_talos_arm_utils STATIC ${CMAKE_SOURCE_DIR}/examples/croc-talos-arm.cpp
                                          ${CMAKE_SOURCE_DIR}/examples/croc-talos-arm.hpp)
  target_include_directories(croc_talos_arm_utils PUBLIC ${CMAKE_SOURCE_DIR}/examples)
  target_link_libraries(croc_talos_arm_utils PUBLIC aligator::croc_compat ${PROJECT_NAME}
                                                    Boost::boost)
  target_add_example_robot_data(croc_talos_arm_utils)

  create_bench("croc-talos-arm.cpp" TRUE)
  target_add_example_robot_data(bench-croc-talos-arm)
  create_bench("multithreading.cpp" TRUE)
  target_add_example_robot_data(bench-multithreading)
endif()
