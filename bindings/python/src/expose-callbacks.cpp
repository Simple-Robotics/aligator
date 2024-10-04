/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#include "aligator/python/callbacks.hpp"
#include "aligator/helpers/history-callback.hpp"

namespace aligator {
namespace python {

using context::Scalar;

void exposeHistoryCallback() {
  using HistoryCallback = HistoryCallbackTpl<Scalar>;

  bp::scope in_history =
      bp::class_<HistoryCallback, bp::bases<CallbackBase>>(
          "HistoryCallback", "Store the history of solver's variables.",
          bp::init<bool, bool, bool>((bp::arg("self"),
                                      bp::arg("store_pd_vars") = true,
                                      bp::arg("store_values") = true,
                                      bp::arg("store_residuals") = true)))
#define _c(name) def_readonly(#name, &HistoryCallback::name)
          ._c(xs)
          ._c(us)
          ._c(lams)
          ._c(values)
          ._c(merit_values)
          ._c(merit_values)
          ._c(prim_infeas)
          ._c(dual_infeas)
          ._c(inner_crits)
          ._c(al_index)
          ._c(prim_tols)
          ._c(dual_tols);
#undef _c
}

void exposeCallbacks() {
  bp::register_ptr_to_python<shared_ptr<CallbackBase>>();

  bp::class_<CallbackWrapper, boost::noncopyable>("BaseCallback",
                                                  "Base callback for solvers.",
                                                  bp::init<>(bp::args("self")))
      .def("call", bp::pure_virtual(&CallbackWrapper::call),
           bp::args("self", "workspace", "results"));

  exposeHistoryCallback();
}
} // namespace python
} // namespace aligator
