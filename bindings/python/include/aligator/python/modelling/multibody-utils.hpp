/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/modelling/multibody/context.hpp"

namespace aligator {
namespace python {

template <typename Class>
struct FrameAPIVisitor : bp::def_visitor<FrameAPIVisitor<Class>> {

  template <class PyClass> void visit(PyClass &cl) const {
    cl.add_property("frame_id", &Class::getFrameId, &Class::setFrameId,
                    "Get the Pinocchio frame ID.");
  }
};

} // namespace python
} // namespace aligator
