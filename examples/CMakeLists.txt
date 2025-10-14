#
# Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#

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
  create_ex_or_bench(${exfile} ${exname} DEPENDENCIES ${arg_crex_DEPENDENCIES})
endfunction()

if(NOT BUILD_STANDALONE_PYTHON_INTERFACE)
  create_example(clqr.cpp)
  if(BUILD_WITH_PINOCCHIO_SUPPORT)
    create_example(talos-walk.cpp DEPENDENCIES talos_walk_utils)
    create_example(se2-car.cpp)
  endif()

  if(BUILD_CROCODDYL_COMPAT)
    add_project_private_dependency(example-robot-data 4.0.9 REQUIRED)
    create_example(
      talos-arm.cpp
      DEPENDENCIES aligator::croc_compat croc_talos_arm_utils
    )
  endif()
endif()

if(BUILD_PYTHON_INTERFACE)
  file(GLOB EXAMPLES_PY *.py)

  foreach(pyfile ${EXAMPLES_PY})
    get_filename_component(pyfile_name ${pyfile} NAME)
    execute_process(
      COMMAND
        ${CMAKE_COMMAND} -E create_symlink ${pyfile}
        ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name}
    )
  endforeach()
endif()
