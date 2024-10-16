#include "aligator/gar/mem-req.hpp"

#include <boost/test/unit_test.hpp>

using namespace aligator::gar;

BOOST_AUTO_TEST_CASE(mem_req) {
  MemReq req = MemReq(32).addBytes(sizeof(double) * 3 * 3);
  BOOST_CHECK_EQUAL(req.totalBytes(), 96);

  req = MemReq(64).addBytes(sizeof(double) * 3 * 3);
  BOOST_CHECK_EQUAL(req.totalBytes(), 128);
}

BOOST_AUTO_TEST_CASE(array) {
  MemReq req{32};

  req.addArray<double>(3, 4);
  BOOST_CHECK_EQUAL(req.totalBytes(), 96);

  req = MemReq{32}.addArray<double>(3, 1, 2);
  BOOST_CHECK_EQUAL(req.totalBytes(), 64);

  req = MemReq{64}.addArray<double>(3, 1, 2);
  BOOST_CHECK_EQUAL(req.totalBytes(), 64);
}
