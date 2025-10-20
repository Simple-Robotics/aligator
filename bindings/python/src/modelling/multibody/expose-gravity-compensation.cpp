/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/python/fwd.hpp"
#include "aligator/modelling/multibody/fwd.hpp"
#include "aligator/modelling/multibody/gravity-compensation-residual.hpp"

namespace aligator::python {
using context::MatrixXs;
using context::PinModel;
using context::Scalar;
using context::StageFunction;
using context::StageFunctionData;
using GravityCompensationResidual = GravityCompensationResidualTpl<Scalar>;

void exposeGravityCompensation() {
  bp::class_<GravityCompensationResidual, bp::bases<StageFunction>>(
      "GravityCompensationResidual", bp::no_init)
      .def(bp::init<int, const MatrixXs &, const PinModel &>(
          ("self"_a, "ndx", "actuation_matrix", "model")))
      .def(bp::init<int, const PinModel &>(("self"_a, "ndx", "model")))
      .def_readonly("pin_model", &GravityCompensationResidual::pin_model_)
      .def_readonly("actuation_matrix",
                    &GravityCompensationResidual::actuation_matrix_)
      .def_readonly("use_actuation_matrix",
                    &GravityCompensationResidual::use_actuation_matrix)
      .def(PolymorphicMultiBaseVisitor<StageFunction>());

  bp::class_<GravityCompensationResidual::Data, bp::bases<StageFunctionData>>(
      "GravityCompensationData", bp::no_init)
      .def_readonly("pin_data", &GravityCompensationResidual::Data::pin_data_);
}

} // namespace aligator::python
#endif
