#
# Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#

# Create a benchmark
function(create_bench exfile)
  cmake_parse_arguments(arg_ex "CROC" "" "DEPENDENCIES" ${ARGN})
  set(dependencies ${arg_ex_DEPENDENCIES})

  get_filename_component(_exname ${exfile} NAME_WE)
  set(_exname "bench-${_exname}")
  if(${arg_ex_CROC})
    list(APPEND dependencies aligator::croc_compat croc_talos_arm_utils)
  endif()
  create_ex_or_bench(
    ${exfile}
    ${_exname}
    BENCHMARK
    DEPENDENCIES ${dependencies}
  )
  target_include_directories(${_exname} PRIVATE ../examples)
endfunction()

function(create_gar_bench exfile)
  create_bench(${exfile} DEPENDENCIES gar_test_utils)
endfunction()

create_bench(lqr.cpp)
create_bench(se2-car.cpp)
create_bench(talos-walk.cpp DEPENDENCIES talos_walk_utils)
if(BUILD_CROCODDYL_COMPAT)
  create_bench(croc-talos-arm.cpp CROC)
endif()

create_gar_bench(gar-riccati.cpp)
