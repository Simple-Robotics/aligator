/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#include "aligator/gar/dense-riccati.hpp"
#include "aligator/gar/lqr-problem.hpp"

namespace aligator::python {
using namespace gar;
using context::Scalar;
using riccati_base_t = RiccatiSolverBase<Scalar>;
using lqr_t = LQRProblemTpl<context::Scalar>;

void exposeDenseSolver() {

  bp::class_<RiccatiSolverDense<Scalar>, bp::bases<riccati_base_t>,
             boost::noncopyable>("RiccatiSolverDense",
                                 "Stagewise-dense Riccati solver (using "
                                 "stagewise Bunch-Kaufman factorizations).",
                                 bp::no_init)
      .def(bp::init<const lqr_t &>(("self"_a, "problem")));
}

} // namespace aligator::python
