/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#include "aligator/core/filter.hpp"
#include <eigenpy/std-pair.hpp>

namespace aligator {
namespace python {

void exposeFilter() {
  using Filter = FilterTpl<double>;
  
  eigenpy::StdPairConverter<std::pair<double,double>>::registration();
  StdVectorPythonVisitor<std::vector<std::pair<double, double>>>::expose(
      "StdVec_StdPair_double");

  bp::register_ptr_to_python<shared_ptr<Filter>>();
  bp::class_<Filter>(
      "Filter",
      "A pair filter implementation to help make larger steps during optimization.",
      bp::init<double, double, std::size_t>(bp::args("self", "beta", "alpha_min", "max_num_steps")))
      .def("resetFilter", &Filter::resetFilter,
          bp::args("self", "beta", "alpha_min", "max_num_steps"),
          "Reset the filter parameters.")
      .def("run", &Filter::run,
          bp::args("self", "phi", "alpha_try"),
          "Make a step and add pair if step accepted.")
      .def("accept_pair", &Filter::accept_pair,
          bp::args("self", "fpair"),
          "Pair acceptance function.")
      .def_readonly("beta_", &Filter::beta_,
                    "Distance parameter with other pairs in filter.")
      .def_readonly("alpha_min_", &Filter::alpha_min_,
                    "Minimum alpha step.")
      .def_readonly("filter_pairs_", &Filter::filter_pairs_,
                    "Filter pairs vector.");
}

} // namespace python
} // namespace aligator
