#
# Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#

cmake_policy(SET CMP0057 NEW)

# Create an example
function(create_example exfile)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs DEPENDENCIES)

  cmake_parse_arguments(
    arg_crex
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "${PROJECT_NAME}-example-${exname}")
  create_ex_or_bench(False ${exfile} ${exname})
  if(DEFINED arg_crex_DEPENDENCIES)
    message("   > depends on ${arg_crex_DEPENDENCIES}")
    foreach(dep ${arg_crex_DEPENDENCIES})
      target_link_libraries(${exname} PRIVATE ${dep})
    endforeach()
  endif()
endfunction()

set(EXAMPLES_LIST clqr)

if(PROXSUITE_NLP_WITH_PINOCCHIO_SUPPORT)
  list(APPEND EXAMPLES_LIST se2-car)
endif()
if(PINOCCHIO_V3)
  list(APPEND EXAMPLES_LIST talos-walk)
endif()

foreach(filename ${EXAMPLES_LIST})
  create_example(${filename}.cpp)
endforeach()

if(PINOCCHIO_V3)
  target_add_talos_walk(aligator-example-talos-walk)
endif()

if(BUILD_CROCODDYL_COMPAT)
  ADD_PROJECT_PRIVATE_DEPENDENCY(example-robot-data 4.0.9 REQUIRED)
  create_example(
    talos-arm.cpp
    DEPENDENCIES
      aligator::croc_compat
      croc_talos_arm_utils
      example-robot-data::example-robot-data
  )
endif()

file(GLOB EXAMPLES_PY *.py)

foreach(pyfile ${EXAMPLES_PY})
  get_filename_component(pyfile_name ${pyfile} NAME)
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink ${pyfile}
      ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name}
  )
endforeach()
