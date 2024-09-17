/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/modelling/multibody/context.hpp"
#ifdef ALIGATOR_WITH_PINOCCHIO

#include "aligator/python/fwd.hpp"
#include "aligator/python/modelling/multibody-utils.hpp"
#include "aligator/modelling/multibody/fly-high.hpp"
#include "aligator/python/polymorphic-convertible.hpp"

namespace aligator {
namespace python {

using context::MultibodyPhaseSpace;
using context::PinModel;
using context::Scalar;
using context::StageFunction;
using context::StageFunctionData;
using context::UnaryFunction;

void exposeFlyHigh() {
  using FlyHighResidual = FlyHighResidualTpl<Scalar>;
  PolymorphicMultiBaseVisitor<UnaryFunction, StageFunction> unary_visitor;

  bp::class_<FlyHighResidual, bp::bases<UnaryFunction>>(
      "FlyHighResidual",
      "A residual function :math:`r(x) = v_{j,xy} e^{-s z_j}` where :math:`j` "
      "is a given frame index.",
      bp::no_init)
      .def(bp::init<const int, const PinModel &, pinocchio::FrameIndex, Scalar,
                    std::size_t>(
          ("self"_a, "ndx", "model", "frame_id", "slope", "nu")))
      .def(FrameAPIVisitor<FlyHighResidual>())
      .def(unary_visitor)
      .def_readwrite("slope", &FlyHighResidual::slope_,
                     "The slope parameter of the function.");

  bp::class_<FlyHighResidual::Data, bp::bases<StageFunctionData>>(
      "FlyHighResidualData", bp::no_init)
      .def_readonly("ez", &FlyHighResidual::Data::ez)
      .def_readonly("pin_data", &FlyHighResidual::Data::pdata_);
}

} // namespace python
} // namespace aligator

#endif
