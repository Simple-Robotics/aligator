/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "aligator/gar/proximal-riccati.hpp"

namespace aligator::python {
using namespace gar;
using context::Scalar;
using prox_riccati_t = ProximalRiccatiSolver<Scalar>;
using stage_factor_t = StageFactor<Scalar>;
using riccati_base_t = RiccatiSolverBase<Scalar>;
using lqr_t = LqrProblemTpl<context::Scalar>;

void exposeProxRiccati() {

  using value_t = prox_riccati_t::CostToGo;
  bp::class_<value_t>("value_data", bp::no_init)
      .def_readonly("Pmat", &value_t::Pmat)
      .def_readonly("pvec", &value_t::pvec)
      .def_readonly("Vxx", &value_t::Vxx)
      .def_readonly("vx", &value_t::vx)
      .def_readonly("Vxt", &value_t::Vxt)
      .def_readonly("Vtt", &value_t::Vtt)
      .def_readonly("vt", &value_t::vt);

  bp::class_<stage_factor_t>(
      "StageFactor", "Stagewise factor for the generalized Riccati algorithm.",
      bp::no_init)
      .def_readonly("ff", &stage_factor_t::ff)
      .def_readonly("fb", &stage_factor_t::fb)
      .def_readonly("fth", &stage_factor_t::fth)
      .def_readonly("kktMat", &stage_factor_t::kktMat)
      .def_readonly("kktChol", &stage_factor_t::kktChol)
      .def_readonly("vm", &stage_factor_t::vm);

  using StageFactorVec = std::vector<stage_factor_t>;
  StdVectorPythonVisitor<StageFactorVec, true>::expose("StdVec_StageFactor");

  {
    bp::scope _ =
        bp::class_<prox_riccati_t, bp::bases<riccati_base_t>,
                   boost::noncopyable>("ProximalRiccatiSolver",
                                       "Proximal Riccati solver.", bp::no_init)
            .def(bp::init<const lqr_t &>(("self"_a, "problem")))
            .def_readonly("thGrad", &prox_riccati_t::thGrad, "Value gradient")
            .def_readonly("thHess", &prox_riccati_t::thHess, "Value Hessian")
            .def_readonly("datas", &prox_riccati_t::datas)
            .def_readonly("kkt0", &prox_riccati_t::kkt0,
                          "Initial stage KKT system");
    bp::class_<prox_riccati_t::kkt0_t>("kkt0_t", bp::no_init)
        .def_readonly("ff", &prox_riccati_t::kkt0_t::ff)
        .def_readonly("fth", &prox_riccati_t::kkt0_t::fth)
        .def_readonly("mat", &prox_riccati_t::kkt0_t::mat)
        .def_readonly("chol", &prox_riccati_t::kkt0_t::chol);
  }
}
} // namespace aligator::python
