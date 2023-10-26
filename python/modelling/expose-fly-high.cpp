/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#ifdef PROXDDP_WITH_PINOCCHIO

#include "proxddp/python/fwd.hpp"
#include "proxddp/python/modelling/multibody-utils.hpp"
#include "proxddp/modelling/multibody/fly-high.hpp"

namespace proxddp {
namespace python {

using context::MultibodyPhaseSpace;
using context::Scalar;
using context::StageFunctionData;
using context::UnaryFunction;

void exposeFlyHigh() {
  using FlyHighResidual = FlyHighResidualTpl<Scalar>;
  bp::class_<FlyHighResidual, bp::bases<UnaryFunction>>(
      "FlyHighResidual",
      "A residual function :math:`r(x) = v_{j,xy} e^{-s z_j}` where :math:`j` "
      "is a given frame index.",
      bp::no_init)
      .def(bp::init<shared_ptr<MultibodyPhaseSpace>, pin::FrameIndex, Scalar,
                    std::size_t>(
          bp::args("self", "space", "frame_id", "slope", "nu")))
      .def(FrameAPIVisitor<FlyHighResidual>())
      .def_readwrite("slope", &FlyHighResidual::slope_,
                     "The slope parameter of the function.");

  bp::class_<FlyHighResidual::Data, bp::bases<StageFunctionData>>(
      "FlyHighResidualData", bp::no_init)
      .def_readonly("ez", &FlyHighResidual::Data::ez)
      .def_readonly("pin_data", &FlyHighResidual::Data::pdata_);
}

} // namespace python
} // namespace proxddp

#endif
