#include "proxddp/python/fwd.hpp"
#include "proxddp/python/utils/optional.hpp"

using opt_dbl = boost::optional<double>;

struct mystruct {
  boost::optional<int> a;
  opt_dbl b;
  boost::optional<std::string> msg{"i am struct"};
  mystruct() : a(boost::none), b(boost::none) {}
  mystruct(int a, const opt_dbl &b = boost::none) : a(a), b(b) {}
};

boost::optional<int> none_if_zero(int i) {
  if (i == 0)
    return boost::none;
  else
    return i;
}

boost::optional<mystruct> create_if_true(bool flag, opt_dbl b = boost::none) {
  if (flag) {
    return mystruct(0, b);
  } else {
    return boost::none;
  }
}

BOOST_PYTHON_MODULE(bind_optional) {
  using namespace proxddp::python;
  OptionalConverter<int>::registration();
  OptionalConverter<double>::registration();
  OptionalConverter<std::string>::registration();
  OptionalConverter<mystruct>::registration();

  bp::class_<mystruct>("mystruct", bp::no_init)
      .def(bp::init<>(bp::args("self")))
      .def(bp::init<int, bp::optional<const opt_dbl &>>(
          bp::args("self", "a", "b")))
      .add_property(
          "a",
          bp::make_getter(&mystruct::a,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&mystruct::a))
      .add_property(
          "b",
          bp::make_getter(&mystruct::b,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&mystruct::b))
      .add_property(
          "msg",
          bp::make_getter(&mystruct::msg,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&mystruct::msg));

  bp::def("none_if_zero", none_if_zero, bp::args("i"));
  bp::def("create_if_true", create_if_true, bp::args("flag", "b"));
}
