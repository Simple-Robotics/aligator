set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_NAMES croc-problem croc-action-model)

foreach(name ${TEST_NAMES})
  add_aligator_test(${name})
  set(test_name "test-cpp-${prefix}${name}")
  target_link_libraries(${test_name} PRIVATE aligator::croc_compat)
endforeach(name)
