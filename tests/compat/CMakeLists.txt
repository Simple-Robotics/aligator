set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_NAMES croc-problem croc-action-model)

foreach(name ${TEST_NAMES})
  add_aligator_test2(${name} aligator::croc_compat)
endforeach()
