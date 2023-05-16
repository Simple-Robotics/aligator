/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include <pinocchio/fwd.hpp>
#include "proxddp/python/fwd.hpp"
#include <proxnlp/modelling/spaces/multibody.hpp>

namespace proxddp {

namespace context {

using MultibodyConfiguration = proxnlp::MultibodyConfiguration<Scalar>;
using MultibodyPhaseSpace = proxnlp::MultibodyPhaseSpace<Scalar>;

} // namespace context

namespace python {

template <typename Class>
struct FrameAPIVisitor : bp::def_visitor<FrameAPIVisitor<Class>> {

  template <class PyClass> void visit(PyClass &cl) const {
    cl.add_property("frame_id", &Class::getFrameId, &Class::setFrameId,
                    "Get the Pinocchio frame ID.");
  }
};

} // namespace python
} // namespace proxddp
