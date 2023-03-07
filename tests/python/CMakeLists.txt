#
# Copyright (C) 2023 LAAS-CNRS, INRIA
#

file(GLOB PYTHON_TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test_*.py)
if (NOT BUILD_CROCODDYL_COMPAT)
  list(REMOVE_ITEM PYTHON_TESTS test_compat_croc.py)
endif()
if (NOT BUILD_WITH_PINOCCHIO_PYTHON_BINDINGS)
  list(REMOVE_ITEM PYTHON_TESTS test_constrained_dynamics.py)
  list(REMOVE_ITEM PYTHON_TESTS test_frames.py)
endif()
make_directory(${CMAKE_CURRENT_BINARY_DIR})

foreach(pyfile ${PYTHON_TESTS})
  get_filename_component(pyfile_name ${pyfile} NAME)
  message(STATUS "Adding Python test: ${pyfile_name}")
  ADD_PYTHON_UNIT_TEST(${pyfile_name} "tests/python/${pyfile}" "tests/python")
  set_tests_properties("${pyfile_name}" PROPERTIES DEPENDS ${PYLIB_NAME})
endforeach()
