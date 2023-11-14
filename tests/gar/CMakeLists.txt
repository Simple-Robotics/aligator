set(TEST_NAMES riccati parallel)

foreach(test_name ${TEST_NAMES})
  add_gar_test(${test_name})
endforeach(test_name ${TEST_NAMES})
