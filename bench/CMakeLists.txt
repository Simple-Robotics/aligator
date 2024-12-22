#
# Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#

# Create a benchmark
function(create_bench exfile croc)
  get_filename_component(_exname ${exfile} NAME_WE)
  set(_exname "bench-${_exname}")
  create_ex_or_bench(TRUE ${exfile} ${_exname})

  target_include_directories(${_exname} PRIVATE ../examples)
  target_link_libraries(${_exname} PRIVATE benchmark::benchmark)
  if(croc)
    target_link_libraries(
      ${_exname}
      PRIVATE aligator::croc_compat croc_talos_arm_utils
    )
  endif()
endfunction()

function(create_gar_bench exfile)
  get_filename_component(_exname ${exfile} NAME_WE)
  set(_exname "bench-${_exname}")
  create_ex_or_bench(TRUE ${exfile} ${_exname})

  target_link_libraries(${_exname} PRIVATE benchmark::benchmark gar_test_utils)
endfunction()

create_bench(lqr.cpp FALSE)
create_bench(se2-car.cpp FALSE)
if(BUILD_CROCODDYL_COMPAT)
  create_bench(croc-talos-arm.cpp TRUE)
  target_add_example_robot_data(bench-croc-talos-arm)
endif()
if(PINOCCHIO_V3)
  create_bench(talos-walk.cpp FALSE)
  target_add_example_robot_data(bench-talos-walk)
  target_add_talos_walk(bench-talos-walk)
endif()

create_gar_bench(gar-riccati.cpp)
