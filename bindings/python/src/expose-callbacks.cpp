/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#include "aligator/python/callbacks.hpp"
#include "aligator/helpers/history-callback.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/solvers/fddp/solver-fddp.hpp"

namespace aligator {
namespace python {

using context::Scalar;
using context::SolverFDDP;
using context::SolverProxDDP;
using HistoryCallback = HistoryCallbackTpl<Scalar>;

#define ctor(Solver)                                                           \
  bp::init<Solver *, bool, bool>(                                              \
      ("self"_a, "solver", "store_pd_vars"_a = true, "store_values"_a = true))

void exposeHistoryCallback() {

  bp::scope in_history =
      bp::class_<HistoryCallback, bp::bases<CallbackBase>>(
          "HistoryCallback", "Store the history of solver's variables.",
          bp::no_init)
          .def(ctor(SolverProxDDP))
          .def(ctor(SolverFDDP))
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

  bp::class_<CallbackWrapper, boost::noncopyable>(
      "BaseCallback", "Base callback for solvers.", bp::init<>(("self"_a)))
      .def("call", bp::pure_virtual(&CallbackWrapper::call),
           bp::args("self", "workspace", "results"));

  exposeHistoryCallback();
}
} // namespace python
} // namespace aligator
