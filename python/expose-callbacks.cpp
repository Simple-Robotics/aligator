/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/core/helpers-base.hpp"
#include "proxddp/helpers/history-callback.hpp"

#include <pinocchio/bindings/python/utils/std-vector.hpp>

namespace proxddp {
namespace python {

struct CallbackWrapper : helpers::base_callback<context::Scalar>,
                         bp::wrapper<helpers::base_callback<context::Scalar>> {
  CallbackWrapper() = default;
  void call(const WorkspaceBaseTpl<context::Scalar> &w,
            const ResultsBaseTpl<context::Scalar> &r) {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "call", w, r);
  }
};

void exposeCallbacks() {
  using context::Scalar;
  using callback_t = helpers::base_callback<Scalar>;

  bp::register_ptr_to_python<shared_ptr<callback_t>>();

  bp::class_<CallbackWrapper, shared_ptr<CallbackWrapper>, boost::noncopyable>(
      "BaseCallback", "Base callback for solvers.",
      bp::init<>(bp::args("self")))
      .def("call", bp::pure_virtual(&CallbackWrapper::call),
           bp::args("self", "workspace", "results"));

  {
    using history_storage_t =
        decltype(helpers::history_callback<Scalar>::storage);

    bp::scope in_history =
        bp::class_<helpers::history_callback<Scalar>, bp::bases<callback_t>>(
            "HistoryCallback", "Store the history of solver's variables.",
            bp::init<bool, bool, bool>((bp::arg("self"),
                                        bp::arg("store_pd_vars") = true,
                                        bp::arg("store_values") = true,
                                        bp::arg("store_residuals") = true)))
            .def_readonly("storage",
                          &helpers::history_callback<Scalar>::storage);

    bp::class_<history_storage_t, shared_ptr<history_storage_t>>(
        "history_storage")
        .def_readonly("xs", &history_storage_t::xs)
        .def_readonly("lams", &history_storage_t::lams)
        .def_readonly("values", &history_storage_t::values)
        .def_readonly("merit_values", &history_storage_t::merit_values)
        .def_readonly("prim_infeas", &history_storage_t::prim_infeas)
        .def_readonly("dual_infeas", &history_storage_t::dual_infeas)
        .def_readonly("al_iters", &history_storage_t::al_index)
        .def_readonly("prim_tols", &history_storage_t::prim_tols)
        .def_readonly("dual_tols", &history_storage_t::dual_tols);

    pinocchio::python::StdVectorPythonVisitor<
        std::vector<context::VectorOfVectors>,
        true>::expose("StdVecVec_VectorXs",
                      "std::vector of std::vector of Eigen::MatrixX.");
  }
}
} // namespace python
} // namespace proxddp
